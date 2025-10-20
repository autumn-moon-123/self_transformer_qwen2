import os
import sys
import torch
import argparse
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2-0.5B模型运行器")
    parser.add_argument("--max_tokens", type=int, help="最大生成token数量", default=100)
    parser.add_argument("--prompt", type=str, help="直接输入的提示词", default=None)
    parser.add_argument("--verbose", action="store_true", help="显示调试信息", default=False)
    return parser.parse_args()

# Qwen2 生成类
class Qwen2Generation:
    def __init__(self, generation_config) -> None:
        self.generation_config = generation_config

    def is_token_eos(self, token_id):
        return token_id in self.generation_config.eos_token_id

    def topk_logits_warper(self, logits):
        filter_value = -float("Inf")
        top_k_temp = min(self.generation_config.top_k, logits.size(-1))

        indices_to_remove = logits < torch.topk(logits, top_k_temp)[0][..., -1, None]
        logits_processed = logits.masked_fill(indices_to_remove, filter_value)
        return logits_processed

    def topp_logits_warper(self, logits):
        min_tokens_to_keep = 1
        filter_value = -float("Inf")

        sorted_logits, sorted_indices = torch.sort(logits, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs <= (1 - self.generation_config.top_p)
        sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits_processed = logits.masked_fill(indices_to_remove, filter_value)
        return logits_processed

    def temperature_logits_warper(self, logits):
        logits_processed = logits / self.generation_config.temperature
        return logits_processed

    def repetition_penalty_logits_processor(self, input_ids, logits):
        score = torch.gather(logits, 1, input_ids)
        score = torch.where(
            score < 0,
            score * self.generation_config.repetition_penalty,
            score / self.generation_config.repetition_penalty,
        )
        logits_processed = logits.scatter(1, input_ids, score)
        return logits_processed

    def logits_wrap_process(self, input_ids, logits):
        logits_processed = self.repetition_penalty_logits_processor(input_ids, logits)
        logits_processed = self.temperature_logits_warper(logits_processed)
        logits_processed = self.topk_logits_warper(logits_processed)
        logits_processed = self.topp_logits_warper(logits_processed)
        return logits_processed

    def next_token_id(self, logits, input_ids):
        next_token_logits = self.logits_wrap_process(input_ids, logits.to(torch.float32))
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).squeeze(1)
        return next_token_id
    
# KV缓存类
class KVCache:
    def __init__(self) -> None:
        self.KCache: List[torch.tensor] = []
        self.VCache: List[torch.tensor] = []
        self.verbose = False

    def clear(self):
        self.KCache = []
        self.VCache = []

    def update(self, new_key_states, new_value_states, layer_idx):
        if len(self.KCache) <= layer_idx:
            self.KCache.append(new_key_states)
            self.VCache.append(new_value_states)
        else:
            self.KCache[layer_idx] = torch.cat([self.KCache[layer_idx], new_key_states], dim=-2)
            self.VCache[layer_idx] = torch.cat([self.VCache[layer_idx], new_value_states], dim=-2)
        return self.KCache[layer_idx], self.VCache[layer_idx]

    def get_cached_length(self, layer_idx) -> int:
        if len(self.KCache) <= layer_idx:
            return 0
        return self.KCache[layer_idx].shape[-2]

    def print(self, layer_idx):
        if self.verbose:
            if len(self.KCache) == 0:
                print("缓存为空")
            else:
                print(f"层 {layer_idx} 缓存token数量: ", self.KCache[layer_idx].shape[-2])

# Rope位置编码类
class Rope:
    def __init__(self, hidden_size, max_position_embeddings):
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.sin_matrix, self.cos_matrix = self.matrix

    @property
    def matrix(self):
        seq_list = torch.arange(0, self.hidden_size, 2, dtype=torch.int64).float()
        seq_list = seq_list / self.hidden_size
        seq_list = 10000**seq_list
        theta = 1.0 / seq_list
        t = torch.arange(self.max_position_embeddings, dtype=torch.int64).type_as(theta)
        freqs = torch.outer(t, theta)
        emb = torch.cat((freqs, freqs), dim=-1)
        # 使用float32代替bfloat16，因为CPU对bfloat16支持有限
        cos_val = emb.cos().to(torch.float32)
        sin_val = emb.sin().to(torch.float32)
        return sin_val, cos_val

    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids):
        cos = cos[position_ids]
        sin = sin[position_ids]
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed
    
# RMS归一化类
class RmsNorm:
    def __init__(self, eps):
        self.eps = eps

    def forward(self, states, weights):
        states = states.to(torch.float32)
        variance = states.pow(2).mean(-1, keepdim=True)
        states = states * torch.rsqrt(variance + self.eps)
        return weights * states.to(weights.dtype)

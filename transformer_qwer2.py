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
    
    
# Qwen2模型类
class Qwen2:
    def __init__(self, max_new_tokens, verbose=False):
        self.verbose = verbose
        self.max_new_tokens = max_new_tokens
        self.device = "cpu"  # 强制使用CPU

        # 下载模型
        print("正在加载模型: Qwen2-0.5B-Instruct...")
        self._download_model()
        print(f"模型加载完成! 使用设备: {self.device}")

        # 初始化模型组件
        self.feature_per_head = (int)(self.config.hidden_size / self.config.num_attention_heads)
        self.groups = (int)(self.config.num_attention_heads / self.config.num_key_value_heads)
        self.rope = Rope(self.feature_per_head, self.config.max_position_embeddings)
        self.kv_cache = KVCache()
        self.kv_cache.verbose = verbose
        self.activation = RmsNorm(self.config.rms_norm_eps)

    def _download_model(self):
        try:
            # 明确指定加载到CPU并使用float32
            self.model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2-0.5B-Instruct",
                torch_dtype=torch.float32,  # 使用float32而不是bfloat16
                device_map=None,            # 不使用device_map
                trust_remote_code=True
            ).to("cpu")  # 显式移动到CPU
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            sys.exit(1)

        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        self.config = self.model.config
        generation_config = GenerationConfig.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        self.generation = Qwen2Generation(generation_config)

    def apply_chat_template(self, prompt):
        messages = [
            {"role": "system", "content": "你是一个乐于助人的AI助手"},
            {"role": "user", "content": prompt}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def embedding(self, input: str):
        input_ids = self.tokenizer.encode(input, return_tensors="pt")
        word_embeddings = torch.nn.functional.embedding(input_ids, self.model.model.embed_tokens.weight)
        return input_ids, word_embeddings

    def mlp(self, layer_idx, states):
        gate_proj = torch.nn.functional.linear(states, self.model.model.layers[layer_idx].mlp.gate_proj.weight)
        up_proj = torch.nn.functional.linear(states, self.model.model.layers[layer_idx].mlp.up_proj.weight)
        down_proj = torch.nn.functional.linear(
            torch.nn.functional.silu(gate_proj) * up_proj,
            self.model.model.layers[layer_idx].mlp.down_proj.weight
        )
        return down_proj

    def gqa(self, layer_idx, states, position_id):
        seq_length = states.size()[-2]
        query_states = torch.nn.functional.linear(
            states,
            self.model.model.layers[layer_idx].self_attn.q_proj.weight,
            self.model.model.layers[layer_idx].self_attn.q_proj.bias,
        )

        key_states = torch.nn.functional.linear(
            states,
            self.model.model.layers[layer_idx].self_attn.k_proj.weight,
            self.model.model.layers[layer_idx].self_attn.k_proj.bias,
        )

        value_states = torch.nn.functional.linear(
            states,
            self.model.model.layers[layer_idx].self_attn.v_proj.weight,
            self.model.model.layers[layer_idx].self_attn.v_proj.bias,
        )

        query_states = query_states.view(1, seq_length, self.config.num_attention_heads, self.feature_per_head)

        key_states = key_states.view(1, seq_length, self.config.num_key_value_heads, self.feature_per_head)
        value_states = value_states.view(1, seq_length, self.config.num_key_value_heads, self.feature_per_head)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        kv_cached_seq_len = seq_length + self.kv_cache.get_cached_length(layer_idx)
        cos = self.rope.cos_matrix[:kv_cached_seq_len]
        sin = self.rope.sin_matrix[:kv_cached_seq_len]
        query_states, key_states = self.rope.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_id)

        key_states, value_states = self.kv_cache.update(key_states, value_states, layer_idx)
        key_states = torch.repeat_interleave(key_states, repeats=self.groups, dim=1)
        value_states = torch.repeat_interleave(value_states, repeats=self.groups, dim=1)
        is_causal = seq_length > 1
        attention_out = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, is_causal=is_causal
        )
        attention_out = attention_out.transpose(1, 2)
        attention_out = attention_out.reshape(1, seq_length, self.config.hidden_size)
        attn_out = torch.nn.functional.linear(attention_out, self.model.model.layers[layer_idx].self_attn.o_proj.weight)

        return attn_out

    def decoder_layer(self, layer_idx, states, position_id):
        residual = states
        states = self.activation.forward(states, self.model.model.layers[layer_idx].input_layernorm.weight)
        states = self.gqa(layer_idx, states, position_id)
        states = states + residual

        residual = states
        states = self.activation.forward(states, self.model.model.layers[layer_idx].post_attention_layernorm.weight)
        states = self.mlp(layer_idx, states)
        states = states + residual
        return states

    def module_forward(self, states, position_id):
        for layer_idx in range(self.config.num_hidden_layers):
            states = self.decoder_layer(layer_idx, states, position_id)

        states = self.activation.forward(states, self.model.model.norm.weight)
        return states

    def lm_head(self, states):
        last_hidden_state = states[:, -1, :]
        logits = torch.nn.functional.linear(last_hidden_state, self.model.lm_head.weight)
        return logits

    def generate(self, user_input):
        input_ids = None
        position_id = None
        text_len = 0

        prompt = self.apply_chat_template(user_input)
        answers = ""
        self.kv_cache.clear()

        for _ in range(self.max_new_tokens):
            prompt_ids, embeddings = self.embedding(prompt)
            input_ids = prompt_ids if input_ids is None else input_ids

            if position_id is None:
                text_len = prompt_ids.size()[-1]
                position_id = torch.arange(text_len).reshape(1, text_len)
            else:
                position_id = torch.tensor([[text_len]])
                text_len += 1

            states = self.module_forward(embeddings, position_id)
            logits = self.lm_head(states)
            next_token_id = self.generation.next_token_id(logits, input_ids)
            next_token = self.tokenizer.decode(next_token_id)
            input_ids = torch.cat([input_ids, next_token_id[:, None]], dim=-1)

            prompt = next_token

            if self.generation.is_token_eos(next_token_id):
                break

            answers += next_token

            if self.verbose:
                print(f"下一个词: {next_token} (ID: {next_token_id})")

        return answers
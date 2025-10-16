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
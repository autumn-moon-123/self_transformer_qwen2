import os
import sys
import torch
import argparse
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
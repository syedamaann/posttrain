"""P0: print GPU + tokenizer contract. No training."""
import os
import torch
from transformers import AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-1.7B-Base")
print("gpu :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("vram:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
    print("bf16:", torch.cuda.is_bf16_supported())
print("HF_USER", os.environ.get("HF_USER"), "| HF_TOKEN", bool(os.environ.get("HF_TOKEN")))

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.eos_token = "<|im_end|>"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print("eos", tok.eos_token, tok.eos_token_id, "| pad", tok.pad_token_id)
print("<think>", tok.convert_tokens_to_ids("<think>"),
      "</think>", tok.convert_tokens_to_ids("</think>"),
      "<|endoftext|>", tok.convert_tokens_to_ids("<|endoftext|>"))
assert tok.eos_token_id == 151645
assert tok.convert_tokens_to_ids("<think>") == 151667

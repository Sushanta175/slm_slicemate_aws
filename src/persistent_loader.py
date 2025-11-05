# src/persistent_loader.py -- FP16/BFloat16 only (skip 8-bit attempt)
import gc, traceback, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
OFFLOAD_FOLDER = "offload"

def _load_fp16():
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16  # use bfloat16 on A40; change to torch.float16 if desired
    )
    model.eval()
    return tok, model

print("persistent_loader: loading tokenizer and model (FP16/BFloat16 only)...")
tok = None
model = None
try:
    tok, model = _load_fp16()
    print("persistent_loader: FP16/BFloat16 load succeeded. Model device:", getattr(model, "device", "unknown"))
except Exception:
    print("persistent_loader: FP16 load failed; see traceback:")
    traceback.print_exc()
    tok, model = None, None
    raise

gc.collect()
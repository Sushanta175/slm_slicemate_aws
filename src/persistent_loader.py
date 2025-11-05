# src/persistent_loader.py
import gc
import traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
OFFLOAD_FOLDER = "offload"

def _load_8bit():
    """Try to load with bitsandbytes 8-bit quantization."""
    bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=False)
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=bnb,
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_folder=OFFLOAD_FOLDER,
        offload_buffers=True
    )
    model.eval()
    return tok, model

def _load_fp16():
    """Fallback: load in FP16/BFloat16 (no bitsandbytes)."""
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16  # use bfloat16 on A40; change to float16 if desired
    )
    model.eval()
    return tok, model

print("persistent_loader: loading tokenizer and model (this may take ~10-60s)...")
tok = None
model = None
try:
    try:
        tok, model = _load_8bit()
        print("persistent_loader: 8-bit load succeeded. Model device:", getattr(model, "device", "unknown"))
    except Exception as e:
        print("persistent_loader: 8-bit load failed (falling back to FP16). Error (short):", repr(e))
        traceback.print_exc()
        tok, model = _load_fp16()
        print("persistent_loader: FP16/BFloat16 load succeeded. Model device:", getattr(model, "device", "unknown"))
except Exception:
    print("persistent_loader: failed to load model with both 8-bit and FP16. See traceback:")
    traceback.print_exc()
    tok, model = None, None
    raise

gc.collect()

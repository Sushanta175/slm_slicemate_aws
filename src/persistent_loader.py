# src/persistent_loader.py
import gc
import traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
OFFLOAD_FOLDER = "offload"

print("persistent_loader: loading tokenizer and model (this may take ~10-30s)...")
try:
    bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=False)
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    # NOTE: do not pass torch_dtype when using bitsandbytes quantization_config
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=bnb,
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_folder=OFFLOAD_FOLDER,
        offload_buffers=True
    )
    model.eval()
    print("persistent_loader: model loaded and ready on device:", getattr(model, "device", "unknown"))
except Exception:
    print("persistent_loader: failed to load persistent model; see traceback:")
    traceback.print_exc()
    # Ensure variables exist but cause early failure downstream if used.
    tok, model = None, None
    raise

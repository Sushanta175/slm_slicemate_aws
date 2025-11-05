# src/agent_synthesis.py
import os
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # change if you prefer a different model

def _load_model():
    """Load tokenizer+model into GPU (8-bit quantized where possible)."""
    print("🔹 Loading model for synthesis (8-bit, BF16 preference)...")
    bnb = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=False
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    except Exception as e:
        print(f"⚠️ GPU load failed or quantization not supported ({e}). Falling back to CPU/FP16.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map={"": "cpu"})
    return tokenizer, model

def _truncate_code(code: str, max_chars: int = 15000) -> str:
    if len(code) <= max_chars:
        return code
    return code[-max_chars:]

def synthesize(code: str, line: int, max_new_tokens: int = 512) -> str:
    """Generate a candidate slice for the given line."""
    tokenizer, model = _load_model()
    try:
        code_trunc = _truncate_code(code)
        prompt = (
            f"### Task: Generate only the relevant static program slice (code lines only) "
            f"for the specified line.\n\n### Code:\n{code_trunc}\n\n### Slice for line {line}:\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=False,
                use_cache=True
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    finally:
        # unload
        try:
            del inputs, outputs
        except Exception:
            pass
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    return text

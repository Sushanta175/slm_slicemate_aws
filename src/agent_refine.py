# src/agent_refine.py
import os
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

def _load_model():
    print("🔹 Loading model for refinement (8-bit, BF16 preferred)...")
    bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=False)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    except Exception as e:
        print(f"⚠️ GPU load failed ({e}), falling back to CPU.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map={"": "cpu"})
    return tokenizer, model

def refine(code: str, candidate: str, feedback: str, max_new_tokens: int = 512) -> str:
    tokenizer, model = _load_model()
    try:
        prompt = (
            "### Task: Given the code, the current candidate slice, and verifier feedback, produce a corrected slice.\n"
            "Output only the corrected code lines, no explanation.\n\n"
            f"### Code:\n{code}\n\n"
            f"### Current candidate slice:\n{candidate}\n\n"
            f"### Verifier feedback:\n{feedback}\n\n"
            "### Corrected slice:\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=False,
                use_cache=True
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    finally:
        try:
            del inputs, outputs
        except Exception:
            pass
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
    return text

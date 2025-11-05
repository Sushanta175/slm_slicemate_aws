# src/agent_synthesis.py
import gc
import torch
from model_manager import try_load_persistent, get_mode, get_persistent, load_model_percall, unload_model_percall
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()  # reduce noise

# ensure we attempt persistent load at import time (on RunPod A40 this should succeed)
try_load_persistent()

def _truncate_code(code: str, max_chars: int = 15000) -> str:
    if len(code) <= max_chars:
        return code
    return code[-max_chars:]

def _clean_output(text: str) -> str:
    # if model echoes prompts, try to trim to slice section
    if "### Slice" in text:
        return text.split("### Slice")[-1].strip()
    if "### Slice for line" in text:
        return text.split("### Slice for line")[-1].strip()
    return text.strip()

def synthesize(code: str, line: int, max_new_tokens: int = 512) -> str:
    """
    Public API used by control_loop.py. Internally uses persistent model if available,
    otherwise loads model per call (safe on Colab).
    """
    mode = get_mode()
    code_trunc = _truncate_code(code)
    prompt = (
        f"### Task: Generate only the relevant static program slice (code lines only) "
        f"for the specified line.\n\n### Code:\n{code_trunc}\n\n### Slice for line {line}:\n"
    )

    if mode == "persistent":
        tok, model = get_persistent()
        try:
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.2,
                    do_sample=False,
                    use_cache=True
                )
            text = tok.decode(outputs[0], skip_special_tokens=True)
            return _clean_output(text)
        except RuntimeError as e:
            # fallback to per-call on OOM
            print("agent_synthesis: persistent path OOM/runtime error, switching to per-call for this call:", e)
            try:
                tok2, model2 = load_model_percall()
                inputs = tok2(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model2.device)
                with torch.inference_mode():
                    outputs = model2.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.2, do_sample=False)
                text = tok2.decode(outputs[0], skip_special_tokens=True)
                return _clean_output(text)
            finally:
                try:
                    unload_model_percall(tok2, model2)
                except Exception:
                    pass
        finally:
            # allow GC but do not unload persistent model
            gc.collect()
    else:
        # per-call mode
        tok2, model2 = load_model_percall()
        try:
            inputs = tok2(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model2.device)
            with torch.inference_mode():
                outputs = model2.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.2, do_sample=False)
            text = tok2.decode(outputs[0], skip_special_tokens=True)
            return _clean_output(text)
        finally:
            unload_model_percall(tok2, model2)

# src/agent_refine.py
import gc
import torch
from model_manager import try_load_persistent, get_mode, get_persistent, load_model_percall, unload_model_percall
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

# try persistent at import
try_load_persistent()

def refine(code: str, candidate: str, feedback: str, max_new_tokens: int = 512) -> str:
    """
    Public API used by control_loop.py: returns the corrected slice (string).
    Uses persistent model if available, else per-call.
    """
    mode = get_mode()
    prompt = (
        "### Task: Given the code, the current candidate slice, and verifier feedback, produce a corrected slice.\n"
        "Output only the corrected code lines, no explanation.\n\n"
        f"### Code:\n{code}\n\n"
        f"### Current candidate slice:\n{candidate}\n\n"
        f"### Verifier feedback:\n{feedback}\n\n"
        "### Corrected slice:\n"
    )
    if mode == "persistent":
        tok, model = get_persistent()
        try:
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.3, do_sample=False)
            text = tok.decode(outputs[0], skip_special_tokens=True)
            # trim repeated prompt
            if "### Corrected slice" in text:
                text = text.split("### Corrected slice")[-1].strip()
            return text.strip()
        except RuntimeError as e:
            print("agent_refine: persistent path OOM/runtime error; falling back per-call for this refine():", e)
            try:
                tok2, model2 = load_model_percall()
                inputs = tok2(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model2.device)
                with torch.inference_mode():
                    outputs = model2.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.3, do_sample=False)
                text = tok2.decode(outputs[0], skip_special_tokens=True)
                if "### Corrected slice" in text:
                    text = text.split("### Corrected slice")[-1].strip()
                return text.strip()
            finally:
                try:
                    unload_model_percall(tok2, model2)
                except Exception:
                    pass
        finally:
            gc.collect()
    else:
        tok2, model2 = load_model_percall()
        try:
            inputs = tok2(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model2.device)
            with torch.inference_mode():
                outputs = model2.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.3, do_sample=False)
            text = tok2.decode(outputs[0], skip_special_tokens=True)
            if "### Corrected slice" in text:
                text = text.split("### Corrected slice")[-1].strip()
            return text.strip()
        finally:
            unload_model_percall(tok2, model2)

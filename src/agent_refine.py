# src/agent_refine.py
import gc, torch
from persistent_loader import tok, model
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

def refine(code: str, candidate: str, feedback: str, max_new_tokens: int = 512) -> str:
    """
    Produce a corrected slice given code, current candidate, and verifier feedback.
    """
    if tok is None or model is None:
        raise RuntimeError("Model not loaded (persistent_loader).")
    prompt = (
        "### Task: Given the code, the current candidate slice, and verifier feedback, produce a corrected slice.\n"
        "Output only the corrected code lines, no explanation.\n\n"
        f"### Code:\n{code}\n\n"
        f"### Current candidate slice:\n{candidate}\n\n"
        f"### Verifier feedback:\n{feedback}\n\n"
        "### Corrected slice:\n"
    )
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=False,
            use_cache=True
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    if "### Corrected slice" in text:
        text = text.split("### Corrected slice")[-1].strip()
    gc.collect()
    return text.strip()

# src/agent_synthesis.py
import gc, torch
from persistent_loader import tok, model
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

def _truncate_code(code: str, max_chars: int = 15000) -> str:
    return code if len(code) <= max_chars else code[-max_chars:]

def _clean_output(text: str) -> str:
    if "### Slice" in text:
        return text.split("### Slice")[-1].strip()
    if "### Slice for line" in text:
        return text.split("### Slice for line")[-1].strip()
    return text.strip()

def synthesize(code: str, line: int, max_new_tokens: int = 512) -> str:
    """
    Generate candidate slice for the given code and line.
    Uses the persistent model loaded by persistent_loader.py.
    """
    if tok is None or model is None:
        raise RuntimeError("Model not loaded (persistent_loader).")
    code_trunc = _truncate_code(code)
    prompt = (
        f"### Task: Generate only the relevant static program slice (code lines only) "
        f"for the specified line.\n\n### Code:\n{code_trunc}\n\n### Slice for line {line}:\n"
    )
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=False,
            use_cache=True
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    gc.collect()
    return _clean_output(text)

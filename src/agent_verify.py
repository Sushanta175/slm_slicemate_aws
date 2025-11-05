# src/agent_verify.py
import gc, re, torch
from persistent_loader import tok, model
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

def _clean_lines(text: str) -> str:
    return "\n".join([ln.rstrip() for ln in text.strip().splitlines() if ln.strip()])

def verify(code: str, line: int, candidate: str, gold: list, max_new_tokens: int = 32) -> str:
    """
    Compare candidate to gold; returns one-line verifier string:
    - "OK"
    - "MISSING: <comma-separated line numbers or NONE>; REDUNDANT: <comma-separated line numbers or NONE>"
    The model is instructed to output exactly one line and nothing else.
    """
    if tok is None or model is None:
        raise RuntimeError("Model not loaded (persistent_loader).")
    cand_text = _clean_lines(candidate)
    gold_text = _clean_lines("\n".join(gold)) if gold else ""

    prompt = (
        "### Task: Compare the candidate slice to the gold slice.\n"
        "You MUST reply with exactly ONE LINE and nothing else. The line must be either:\n"
        "  OK\n"
        "OR\n"
        "  MISSING: <comma-separated line numbers or NONE>; REDUNDANT: <comma-separated line numbers or NONE>\n"
        "Do not include any code, explanations, or extra text.\n\n"
        f"### Candidate slice for line {line}:\n{cand_text}\n\n"
        f"### Gold slice:\n{gold_text}\n\n"
        "### Response (one line only):\n"
    )

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            use_cache=True
        )
    text = tok.decode(out[0], skip_special_tokens=True).strip().splitlines()[0].strip()
    gc.collect()
    return text

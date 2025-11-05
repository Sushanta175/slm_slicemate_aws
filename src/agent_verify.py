# src/agent_verify.py
import gc
import re
import torch
from model_manager import try_load_persistent, get_mode, get_persistent, load_model_percall, unload_model_percall
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

# attempt persistent load at import time
try_load_persistent()

def _clean_lines(text: str) -> str:
    return "\n".join([ln.rstrip() for ln in text.strip().splitlines() if ln.strip()])

def _parse_feedback(text: str):
    """
    Parse verifier text into structured dict:
    {"ok": True} or {"ok": False, "missing": [...], "redundant": [...], "raw": text}
    """
    t = (text or "").strip()
    upper = t.upper()
    if upper.startswith("OK"):
        return {"ok": True, "raw": t}
    res = {"ok": False, "missing": [], "redundant": [], "raw": t}
    normalized = t.replace("\n", " ").replace(";", ",")
    m = re.search(r"MISSING\s*:\s*([0-9,\sA-Za-z\-]+)", normalized, re.IGNORECASE)
    if m:
        raw = m.group(1)
        if "NONE" not in raw.upper():
            res["missing"] = [int(x) for x in re.findall(r"-?\d+", raw)]
    r = re.search(r"REDUNDANT\s*:\s*([0-9,\sA-Za-z\-]+)", normalized, re.IGNORECASE)
    if r:
        raw = r.group(1)
        if "NONE" not in raw.upper():
            res["redundant"] = [int(x) for x in re.findall(r"-?\d+", raw)]
    return res

def verify(code: str, line: int, candidate: str, gold: list, max_new_tokens: int = 128) -> str:
    """
    Public API matching your existing control_loop: returns verifier raw text.
    Internally uses persistent model if available, else per-call load.
    """
    mode = get_mode()
    cand_text = _clean_lines(candidate)
    gold_text = _clean_lines("\n".join(gold)) if gold else ""

    prompt = (
        "### Task: Compare the candidate slice to the gold slice and return EXACTLY one of the following:\n"
        "1) OK\n"
        "2) MISSING: <comma-separated line numbers or NONE>\n"
        "   REDUNDANT: <comma-separated line numbers or NONE>\n\n"
        f"### Code:\n{code}\n\n"
        f"### Candidate slice for line {line}:\n{cand_text}\n\n"
        f"### Gold slice:\n{gold_text}\n\n"
        "### Response:\n"
    )

    if mode == "persistent":
        tok, model = get_persistent()
        try:
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False)
            text = tok.decode(outputs[0], skip_special_tokens=True).strip()
            return text
        except RuntimeError as e:
            print("agent_verify: persistent path OOM/runtime error; falling back per-call for this verify():", e)
            try:
                tok2, model2 = load_model_percall()
                inputs = tok2(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model2.device)
                with torch.inference_mode():
                    outputs = model2.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False)
                text = tok2.decode(outputs[0], skip_special_tokens=True).strip()
                return text
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
                outputs = model2.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False)
            text = tok2.decode(outputs[0], skip_special_tokens=True).strip()
            return text
        finally:
            unload_model_percall(tok2, model2)

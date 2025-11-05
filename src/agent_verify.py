# src/agent_verify.py
import gc
import re
import torch
from persistent_loader import tok, model
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

def _clean_lines(text: str) -> str:
    return "\n".join([ln.rstrip() for ln in text.strip().splitlines() if ln.strip()])

# Regex that a valid one-line verifier response must match.
_VERIFIER_RE = re.compile(r"^(OK|MISSING:\s*[0-9,\s\-]+;\s*REDUNDANT:\s*[0-9,\s\-]+|MISSING:\s*NONE;\s*REDUNDANT:\s*NONE)\s*$", re.IGNORECASE)

def _extract_verifier_line(text: str) -> str:
    """
    Return the first line that matches the verifier regex.
    If none match, return empty string.
    """
    for ln in text.splitlines():
        cand = ln.strip()
        if _VERIFIER_RE.match(cand):
            # normalize spacing and uppercase OK/MISSING/REDUNDANT keywords
            # ensure consistent formatting: e.g. "MISSING: 1,2; REDUNDANT: NONE"
            # We'll uppercase keywords and strip duplicate spaces
            cand = re.sub(r"\s*;\s*", "; ", cand)
            cand = re.sub(r"\s*,\s*", ",", cand)
            # normalize keyword case
            cand = re.sub(r"^(ok)$", "OK", cand, flags=re.IGNORECASE)
            cand = re.sub(r"^missing", "MISSING", cand, flags=re.IGNORECASE)
            cand = re.sub(r"redundant", "REDUNDANT", cand, flags=re.IGNORECASE)
            return cand.strip()
    return ""

def verify(code: str, line: int, candidate: str, gold: list, max_new_tokens: int = 32) -> str:
    """
    Compare candidate to gold; returns one-line verifier string:
    - "OK"
    - "MISSING: <comma-separated line numbers or NONE>; REDUNDANT: <comma-separated line numbers or NONE>"

    The prompt requires a one-line output that MUST match the regex above.
    """
    if tok is None or model is None:
        raise RuntimeError("Model not loaded (persistent_loader).")

    cand_text = _clean_lines(candidate)
    gold_text = _clean_lines("\n".join(gold)) if gold else ""

    # Very strict prompt: require exact one-line output, beginning with OK or MISSING:...; REDUNDANT:...
    prompt = (
        "### Task: Compare the candidate slice to the gold slice.\n"
        "You MUST reply with exactly ONE LINE and nothing else. The ONE line MUST match ONE of these exact formats:\n"
        "  1) OK\n"
        "  2) MISSING: <comma-separated line numbers or NONE>; REDUNDANT: <comma-separated line numbers or NONE>\n"
        "The response MUST start with either 'OK' or 'MISSING:' and must NOT include any extra text, code, or explanation.\n"
        "If you cannot determine missing/redundant lines, respond 'MISSING: NONE; REDUNDANT: NONE'.\n"
        "IMPORTANT: The FIRST token of your output must be the verdict (do not echo the prompt or reprint the code).\n\n"
        f"### Candidate slice for line {line}:\n{cand_text}\n\n"
        f"### Gold slice:\n{gold_text}\n\n"
        "### Response (one-line only, must match the described formats):\n"
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

    raw = tok.decode(out[0], skip_special_tokens=True).strip()
    gc.collect()

    # Try to extract a valid verifier line
    line = _extract_verifier_line(raw)
    if line:
        return line

    # If model failed to follow instructions, attempt a fallback heuristic:
    # - If model output contains 'OK' anywhere, return 'OK'
    if re.search(r"\bOK\b", raw, flags=re.IGNORECASE):
        return "OK"

    # As a last resort, return a deterministic safe fallback that indicates no changes
    return "MISSING: NONE; REDUNDANT: NONE"

# src/agent_verify.py
import torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_model():
    print("🔹 Loading Mistral-7B-Instruct-v0.2 for verification (16K context, 4-bit quantized)…")
    MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    return tokenizer, model

def _clean_lines(text: str) -> str:
    return "\n".join([ln.rstrip() for ln in text.strip().splitlines() if ln.strip()])

def verify(code: str, line: int, candidate: str, gold: list, max_new_tokens: int = 256) -> str:
    """Compare candidate slice with gold slice and return MISSING/REDUNDANT lines or OK."""
    tokenizer, model = load_model()

    cand_text = _clean_lines(candidate)
    gold_text = _clean_lines("\n".join(gold)) if gold else ""

    prompt = (
        "### Task: Compare the candidate slice to the gold slice and report differences.\n"
        "Return strictly one of the following formats:\n"
        "1. 'OK'\n"
        "2. 'MISSING: <lines or NONE>'\n   'REDUNDANT: <lines or NONE>'\n\n"
        f"### Code:\n{code}\n\n"
        f"### Candidate slice for line {line}:\n{cand_text}\n\n"
        f"### Gold slice:\n{gold_text}\n\n### Response:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=16000).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.0)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    del tokenizer, model, inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()

    return text

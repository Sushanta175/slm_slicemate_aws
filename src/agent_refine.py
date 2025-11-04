# src/agent_refine.py
import torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_model():
    print("🔹 Loading Mistral-7B-Instruct-v0.2 for refinement (16K context, 4-bit quantized)…")
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

def refine(code: str, candidate: str, feedback: str, max_new_tokens: int = 512) -> str:
    """Use verifier feedback to refine the candidate slice."""
    tokenizer, model = load_model()

    prompt = (
        "### Task: Given the code, current slice, and verifier feedback, produce a corrected slice. "
        "Output only valid code lines, with no explanation.\n\n"
        f"### Code:\n{code}\n\n"
        f"### Current candidate slice:\n{candidate}\n\n"
        f"### Verifier feedback:\n{feedback}\n\n"
        "### Corrected slice:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=16000).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.3)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    del tokenizer, model, inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()

    return text

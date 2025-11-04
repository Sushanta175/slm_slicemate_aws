# src/agent_synthesis.py
import torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_model():
    print("🔹 Loading Mistral-7B-Instruct-v0.2 for synthesis (16K context, 4-bit quantized)…")
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

def _truncate_code(code: str, max_chars: int = 15000) -> str:
    return code if len(code) <= max_chars else code[-max_chars:]

def synthesize(code: str, line: int, max_new_tokens: int = 512) -> str:
    """Generate candidate slice for the given line using Mistral."""
    tokenizer, model = load_model()

    code_trunc = _truncate_code(code)
    prompt = (
        f"### Task: Generate only the relevant static program slice (code lines only) "
        f"for the specified line.\n\n### Code:\n{code_trunc}\n\n### Slice for line {line}:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=16000).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.2)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # 🧹 Cleanup
    del tokenizer, model, inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()

    return text

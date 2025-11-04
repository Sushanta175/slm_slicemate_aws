import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from eval.metrics import f1_exact

base_model = "microsoft/Phi-3-mini-4k-instruct"
adapter  = "./adapter_phi3"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16
).cuda()
model = PeftModel.from_pretrained(model, adapter)
model.eval()

data = [json.loads(l) for l in open("data/mini_slicebench.jsonl")]  # original 50
f1s = []

for i, ex in enumerate(data[:50]):
    print(f"Processing {i+1}/50")
    code, line_no, gold = ex["code"], ex["line"], ex["slice"]
    prompt = f"### Code:\n{code}\n### Slice for line {line_no}:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, temperature=0, do_sample=False)
    pred_raw = tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    pred = [l.strip() for l in pred_raw.splitlines() if l.strip()]
    _, _, f1 = f1_exact(pred, gold)
    f1s.append(f1)

avg = sum(f1s) / len(f1s)
print(f"Phi-3-LoRA F1 (50 samples) = {avg:.3f}")
with open("LOG.md", "a") as f:
    f.write(f"Phi-3-LoRA F1 = {avg:.3f}\n")
import json, os, sys
from openai import OpenAI
sys.path.append(".")
from eval.metrics import f1_exact

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

data = [json.loads(l) for l in open("data/mini_slicebench.jsonl")]
f1s = []

for ex in data:
    code, line_no, gold = ex["code"], ex["line"], ex["slice"]
    prompt = f"""You are an expert at static slicing.
Return ONLY the lines (in original order) that influence line {line_no}.
Do not add explanations.

Code:
{code}
"""
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    pred = resp.choices[0].message.content.strip().splitlines()
    _, _, f1 = f1_exact(pred, gold)
    f1s.append(f1)

avg = sum(f1s) / len(f1s)
print(f"GPT-4 baseline F1 (50 samples) = {avg:.3f}")
with open("LOG.md", "a") as f:
    f.write(f"GPT-4 baseline F1 = {avg:.3f}\n")
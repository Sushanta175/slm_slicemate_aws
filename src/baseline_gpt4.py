import json, random, os, sys
sys.path.append(".")
from eval.metrics import f1_exact
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

path = "data/slicebench_python.jsonl"
data = [json.loads(l) for l in open(path)]
sample = random.sample(data, 50)          # reproducible: random.seed(42)

def ask_gpt4(code, line):
    prompt = f"""You are an AI that performs backward static slicing.
Return ONLY the lines (in original order) that influence line {line}.
Do not add explanations.

Code:
{code}
"""
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip().splitlines()

results = []
for ex in sample:
    pred = ask_gpt4(ex["code"], ex["line"])
    gold = ex["slice"]
    p, r, f1 = f1_exact(pred, gold)
    results.append(f1)

avg_f1 = sum(results) / len(results)
print(f"GPT-4 mean F1 (50 samples) = {avg_f1:.3f}")
with open("LOG.md", "a") as f:
    f.write(f"\nWeek-0 GPT-4 baseline F1 = {avg_f1:.3f}\n")
# src/02_annotate_with_gpt4.py
import json, os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

inp = open("data/mini_slicebench_raw.jsonl", encoding="utf8")
out = open("data/mini_slicebench.jsonl", "w", encoding="utf8")

for line in inp:
    ex = json.loads(line)
    code, line_no = ex["code"], ex["line"]
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
    slice_lines = resp.choices[0].message.content.strip().splitlines()
    ex["slice"] = slice_lines
    out.write(json.dumps(ex) + "\n")

inp.close()
out.close()
print("Gold slices saved → data/mini_slicebench.jsonl")
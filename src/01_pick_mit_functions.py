import json, os, re, logging
logging.disable(logging.WARNING)   # shut HF up

from datasets import load_dataset

os.makedirs("data", exist_ok=True)
out = open("data/mini_slicebench_raw.jsonl", "w", encoding="utf8")

ds = load_dataset("bigcode/the-stack-dedup", streaming=True, split="train")
count = 0

for row in ds:
    code = row["content"]          # raw file content
    # quick heuristic: single function, 15-50 lines, contains print()
    lines = code.splitlines()
    if 15 <= len(lines) <= 50 and "def " in code and any("print(" in l for l in lines):
        # pick first print() line as slicing criterion
        for i, l in enumerate(lines):
            if "print(" in l:
                item = {"code": code, "line": i + 1, "slice": []}
                out.write(json.dumps(item) + "\n")
                count += 1
                print(f"Collected {count}/50")
                break
    if count == 400:
        break

out.close()
print("Saved → data/mini_slicebench_raw.jsonl")
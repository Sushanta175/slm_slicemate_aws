\# src/control_loop.py
import json
import os
import torch, gc
from agent_synthesis import synthesize
from agent_verify import verify
from agent_refine import refine

MAX_ITERS = 3
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "mini_slicebench_final.jsonl")
RESULT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "slm_slice.jsonl")

def clean_gpu():
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

def control_loop():
    print("🔹 Starting multi-agent slice synthesis pipeline…")
    examples = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except Exception:
                continue

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    results = []
    ok_count = 0
    total = len(examples)

    for idx, ex in enumerate(examples[:50]):  # limit adjustable
        code, line = ex.get("code", ""), ex.get("line", 0)
        print(f"\n=== Example {idx+1} Line {line} ===")

        candidate = synthesize(code, line)
        print("\nInitial Slice (head):\n", candidate[:1200])
        clean_gpu()

        feedback = ""
        for it in range(MAX_ITERS):
            feedback = verify(code, line, candidate, ex.get("gold", []))
            print(f"\nIteration {it+1} Feedback:\n{feedback}")
            clean_gpu()

            if feedback.strip().upper().startswith("OK"):
                print("✅ Slice verified as correct")
                ok_count += 1
                break

            candidate = refine(code, candidate, feedback)
            print("🔁 Refined Slice (head):\n", candidate[:1200])
            clean_gpu()

        results.append({
            "id": ex.get("id", idx),
            "line": line,
            "final_slice": candidate,
            "feedback": feedback,
        })

        # write incremental progress
        with open(RESULT_PATH, "w", encoding="utf-8") as fout:
            for r in results:
                fout.write(json.dumps(r) + "\n")

    try:
        print(f"\n🏁 Done. Accuracy: {ok_count}/{total} = {ok_count/total:.2%}")
    except ZeroDivisionError:
        print("\n⚠️ No valid examples processed.")

if __name__ == "__main__":
    control_loop()

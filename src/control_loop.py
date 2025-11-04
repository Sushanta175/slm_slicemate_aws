# src/control_loop.py
import json, os, torch, gc
from agent_synthesis import synthesize
from agent_verify import verify
from agent_refine import refine

MAX_ITERS = 3
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "mini_slicebench_final.jsonl")
RESULT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "slm_slice.jsonl")

def clean_gpu():
    torch.cuda.empty_cache()
    gc.collect()

def control_loop():
    print("🔹 Starting multi-agent slice synthesis pipeline…")
    examples = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                examples.append(json.loads(line))
            except:
                continue

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    results, ok_count, total = [], 0, len(examples)

    for idx, ex in enumerate(examples[:20]):
        code, line = ex.get("code", ""), ex.get("line", 0)
        print(f"\n=== Example {idx+1} Line {line} ===")

        candidate = synthesize(code, line)
        print("\nInitial Slice:\n", candidate[:1000])
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
            print("🔁 Refined Slice:\n", candidate[:1000])
            clean_gpu()

        results.append({
            "id": ex.get("id", idx),
            "line": line,
            "final_slice": candidate,
            "feedback": feedback,
        })

        with open(RESULT_PATH, "w", encoding="utf-8") as fout:
            for r in results:
                fout.write(json.dumps(r) + "\n")

    print(f"\n🏁 Done. Accuracy: {ok_count}/{total} = {ok_count/total:.2%}")

if __name__ == "__main__":
    control_loop()

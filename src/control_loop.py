# src/control_loop.py
import json
import os
import time
import gc
import traceback

import torch

from model_manager import get_mode  # to show persistent/percall
from agent_synthesis import synthesize
from agent_verify import verify
from agent_refine import refine

# optional: import your metric function (assumes eval/metrics.py exists)
try:
    from eval.metrics import f1_exact
except Exception:
    # fallback simple set-based F1 if eval.metrics isn't available
    def f1_exact(pred_lines, gold_lines):
        pred, gold = set([ln.strip() for ln in pred_lines if ln.strip()]), set([ln.strip() for ln in gold_lines if ln.strip()])
        tp = len(pred & gold)
        fp = len(pred - gold)
        fn = len(gold - pred)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f1

MAX_ITERS = 3
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "mini_slicebench_final.jsonl")
RESULT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "slm_slice.jsonl")

def clean_gpu():
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

def control_loop(limit: int = 50):
    print("🔹 Starting multi-agent slice synthesis pipeline…")
    mode = get_mode()
    print(f"ℹ️ Model manager mode: {mode}")

    # load examples
    examples = []
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
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
    f1_sum = 0.0
    processed = 0
    total = min(len(examples), limit)

    start_all = time.time()
    for idx, ex in enumerate(examples[:limit]):
        example_start = time.time()
        try:
            code = ex.get("code", "")
            line = ex.get("line", 0)
            gold = ex.get("gold", ex.get("slice", [])) or []
            ex_id = ex.get("id", idx)
            print(f"\n=== Example {idx+1}/{total} id={ex_id} line={line} ===")

            # Synthesis
            t0 = time.time()
            candidate = synthesize(code, line)
            t_synth = time.time() - t0
            print(f"[synth] time={t_synth:.2f}s len={len(candidate):,}")

            # Verification + refinement loop
            feedback_text = ""
            verified_ok = False
            t_verify_total = 0.0
            t_refine_total = 0.0
            for it in range(MAX_ITERS):
                t0 = time.time()
                fb = verify(code, line, candidate, gold)
                t_v = time.time() - t0
                t_verify_total += t_v
                print(f"[verify] iter={it+1} time={t_v:.2f}s -> {str(fb)[:300]!r}")
                feedback_text = fb

                # treat "OK" (case-insensitive) as success
                if isinstance(fb, str) and fb.strip().upper().startswith("OK"):
                    verified_ok = True
                    ok_count += 1
                    print("✅ Verified OK")
                    break

                # refinement
                t0 = time.time()
                candidate = refine(code, candidate, fb)
                t_r = time.time() - t0
                t_refine_total += t_r
                print(f"[refine] iter={it+1} time={t_r:.2f}s len={len(candidate):,}")

            # compute F1 vs gold (line text-based)
            pred_lines = [ln for ln in candidate.splitlines() if ln.strip()]
            gold_lines = [ln for ln in gold if str(ln).strip()]
            try:
                prec, rec, f1 = f1_exact(pred_lines, gold_lines)
            except Exception:
                prec, rec, f1 = 0.0, 0.0, 0.0
            f1_sum += f1
            processed += 1

            example_time = time.time() - example_start
            print(f"[example done] id={ex_id} synth={t_synth:.2f}s verify={t_verify_total:.2f}s refine={t_refine_total:.2f}s total={example_time:.2f}s f1={f1:.4f}")

            # prepare result record
            rec = {
                "id": ex_id,
                "line": line,
                "final_slice": candidate,
                "verified_ok": verified_ok,
                "feedback": feedback_text,
                "f1": f1,
                "timings": {
                    "synth": t_synth,
                    "verify_total": t_verify_total,
                    "refine_total": t_refine_total,
                    "example_total": example_time
                }
            }
            results.append(rec)

            # append to results file (safe append)
            with open(RESULT_PATH, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(rec) + "\n")

            # light cleanup between examples
            clean_gpu()
            time.sleep(0.1)

        except Exception as e:
            print("❌ Error processing example", idx, "id=", ex.get("id", idx))
            traceback.print_exc()
            # still append a minimal failure record
            fail_rec = {"id": ex.get("id", idx), "error": str(e)}
            with open(RESULT_PATH, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(fail_rec) + "\n")
            clean_gpu()
            continue

    total_time = time.time() - start_all
    avg_f1 = (f1_sum / processed) if processed else 0.0
    print(f"\n🏁 Done. Processed {processed}/{total} examples.")
    print(f"Verified OK: {ok_count}/{processed} ({(ok_count/processed*100) if processed else 0:.2f}%)")
    print(f"Average F1 over processed examples: {avg_f1:.4f}")
    print(f"Total pipeline time: {total_time:.1f}s (avg {total_time/processed:.2f}s per example)" if processed else f"Total pipeline time: {total_time:.1f}s")
    return results

if __name__ == "__main__":
    control_loop(limit=50)

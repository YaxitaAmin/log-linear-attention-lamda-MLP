import subprocess
import itertools
import os
import json
from datetime import datetime

# sweep config
LAMBDA_MODES  = ["fixed", "mlp_softplus", "mlp_softmax"]
KV_PAIRS      = [4, 8, 16, 32]
SEQ_LENS      = [256, 512]
SEEDS         = [0, 1, 2]
MODEL_DIM     = 64
BATCH_SIZE    = 64
MAX_STEPS     = 5000
LR            = 1e-3
MLP_HIDDEN    = 64
OUTPUT_DIR    = "results/mqar"

def run_already_done(lambda_mode, num_kv_pairs, seq_len, seed):
    run_name = f"{lambda_mode}_kv{num_kv_pairs}_seq{seq_len}_seed{seed}"
    result_path = os.path.join(OUTPUT_DIR, run_name, "result.json")
    return os.path.exists(result_path)

def main():
    combos = list(itertools.product(LAMBDA_MODES, KV_PAIRS, SEQ_LENS, SEEDS))
    total  = len(combos)
    print(f"Total runs: {total}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    summary = []
    for idx, (mode, kv, seq, seed) in enumerate(combos, 1):
        run_name = f"{mode}_kv{kv}_seq{seq}_seed{seed}"
        print(f"\n[{idx}/{total}] {run_name}")

        # skip if already done
        if run_already_done(mode, kv, seq, seed):
            print(f"  -> Already done, skipping!")
            result_path = os.path.join(OUTPUT_DIR, run_name, "result.json")
            with open(result_path) as f:
                r = json.load(f)
            summary.append({"run": run_name, "best_val_acc": r["best_val_acc"], "status": "skipped"})
            continue

        cmd = [
            "python", "train_mqar.py",
            "--lambda_mode",    mode,
            "--model_dim",      str(MODEL_DIM),
            "--num_kv_pairs",   str(kv),
            "--seq_len",        str(seq),
            "--batch_size",     str(BATCH_SIZE),
            "--max_steps",      str(MAX_STEPS),
            "--lr",             str(LR),
            "--mlp_hidden_dim", str(MLP_HIDDEN),
            "--seed",           str(seed),
            "--output_dir",     OUTPUT_DIR,
        ]

        try:
            result = subprocess.run(cmd, check=True)
            # read result json
            result_path = os.path.join(OUTPUT_DIR, run_name, "result.json")
            if os.path.exists(result_path):
                with open(result_path) as f:
                    r = json.load(f)
                best_acc = r["best_val_acc"]
                summary.append({"run": run_name, "best_val_acc": best_acc, "status": "done"})
                print(f"  -> Finished! Best acc: {best_acc:.1f}%")
            else:
                summary.append({"run": run_name, "best_val_acc": None, "status": "no_result"})
        except subprocess.CalledProcessError as e:
            print(f"  -> Failed! Error: {e}")
            summary.append({"run": run_name, "best_val_acc": None, "status": "failed"})

    # save master summary
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Sweep done! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Summary saved to {summary_path}")
    print(f"{'='*60}\n")

    # print quick leaderboard
    done = [s for s in summary if s["best_val_acc"] is not None]
    done.sort(key=lambda x: x["best_val_acc"], reverse=True)
    print("Top 10 runs:")
    for s in done[:10]:
        print(f"  {s['best_val_acc']:6.1f}%  {s['run']}")

if __name__ == "__main__":
    main()

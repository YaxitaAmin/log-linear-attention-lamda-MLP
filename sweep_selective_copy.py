"""
sweep_selective_copy.py  —  Sweep all configs for selective copying benchmark

Runs all combinations of lambda_mode x seq_len x num_tokens x seeds
on your A100. Should complete in a few hours!

Usage:
    python sweep_selective_copy.py
"""

import subprocess
import itertools
import os
import json
from datetime import datetime

# ── sweep config ──────────────────────────────────────────────────────────────
LAMBDA_MODES = ["fixed", "mlp_softplus", "mlp_softmax"]
SEQ_LENS     = [256, 512]
NUM_TOKENS   = [16]        # 16 tokens to copy is standard
SEEDS = [0, 1, 2, 3, 4] #adding 2 more additional seeds
MODEL_DIM    = 64
BATCH_SIZE   = 64
MAX_STEPS    = 30000
LR           = 1e-3
MLP_HIDDEN   = 64
OUTPUT_DIR   = "results/selective_copy"


def already_done(lambda_mode, num_tokens, seq_len, seed):
    run_name    = f"{lambda_mode}_tok{num_tokens}_seq{seq_len}_seed{seed}"
    result_path = os.path.join(OUTPUT_DIR, run_name, "result.json")
    return os.path.exists(result_path)


def main():
    combos = list(itertools.product(LAMBDA_MODES, SEQ_LENS, NUM_TOKENS, SEEDS))
    total  = len(combos)
    print(f"Total runs : {total}")
    print(f"Started at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    summary = []
    for idx, (mode, seq, tok, seed) in enumerate(combos, 1):
        run_name = f"{mode}_tok{tok}_seq{seq}_seed{seed}"
        print(f"\n[{idx}/{total}] {run_name}")

        if already_done(mode, tok, seq, seed):
            print(f"  -> Already done, skipping!")
            result_path = os.path.join(OUTPUT_DIR, run_name, "result.json")
            with open(result_path) as f:
                r = json.load(f)
            summary.append({"run": run_name, "best_val_acc": r["best_val_acc"], "status": "skipped"})
            continue

        cmd = [
            "python", "train_selective_copy.py",
            "--lambda_mode",    mode,
            "--model_dim",      str(MODEL_DIM),
            "--num_tokens",     str(tok),
            "--seq_len",        str(seq),
            "--batch_size",     str(BATCH_SIZE),
            "--max_steps",      str(MAX_STEPS),
            "--lr",             str(LR),
            "--mlp_hidden_dim", str(MLP_HIDDEN),
            "--seed",           str(seed),
            "--output_dir",     OUTPUT_DIR,
        ]

        try:
            subprocess.run(cmd, check=True)
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

    # save summary
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Sweep done! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Summary -> {summary_path}")
    print(f"{'='*60}\n")

    # print results table
    print(f"{'Mode':<20} {'seq':<8} {'tok':<6} {'seed':<6} {'acc':>8}")
    print("-" * 52)
    for s in sorted(summary, key=lambda x: x["run"]):
        if s["best_val_acc"] is not None:
            parts = s["run"].split("_")
            print(f"{parts[0]:<20} {parts[2][3:]:<8} {parts[1][3:]:<6} {parts[3][4:]:<6} {s['best_val_acc']:>7.1f}%")

    # summary table per mode x seq
    print(f"\n--- Mean accuracy per mode x seq_len (averaged over seeds & tokens) ---")
    from collections import defaultdict
    acc_store = defaultdict(list)
    for s in summary:
        if s["best_val_acc"] is not None:
            parts    = s["run"].split("_")
            mode     = parts[0]
            seq      = parts[2][3:]
            acc_store[(mode, seq)].append(s["best_val_acc"])

    modes = LAMBDA_MODES
    seqs  = [str(s) for s in SEQ_LENS]
    header = f"{'Mode':<20}" + "".join(f"  seq={s:<6}" for s in seqs)
    print(header)
    print("-" * len(header))
    for mode in modes:
        row = f"{mode:<20}"
        for seq in seqs:
            vals = acc_store.get((mode, seq), [])
            if vals:
                row += f"  {sum(vals)/len(vals):>7.1f}%"
            else:
                row += f"  {'n/a':>8}"
        print(row)


if __name__ == "__main__":
    main()
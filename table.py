import json
import csv
from collections import defaultdict

# load data
with open("results/mqar/sweep_summary.json") as f:
    runs = json.load(f)

# parse run names and filter out seed2
results = defaultdict(list)

for run in runs:
    name = run["run"]
    parts = name.split("_")

    # handle prefix: fixed vs mlp_softplus vs mlp_softmax
    if parts[0] == "fixed":
        mode = "fixed"
        rest = parts[1:]
    elif parts[0] == "mlp":
        mode = f"mlp_{parts[1]}"  # mlp_softplus or mlp_softmax
        rest = parts[2:]
    else:
        continue

    # rest = [kvX, seqY, seedZ]
    kv = int(rest[0].replace("kv", ""))
    seq = int(rest[1].replace("seq", ""))
    seed = int(rest[2].replace("seed", ""))

    # drop seed2
    if seed == 2:
        continue

    key = (mode, kv, seq)
    results[key].append(run["best_val_acc"])

# compute mean over seed0 and seed1
summary = {}
for (mode, kv, seq), accs in results.items():
    summary[(mode, kv, seq)] = round(sum(accs) / len(accs), 2)

# ------- print table: focus on seq=256 + highlight kv32 seq=512 -------
modes = ["fixed", "mlp_softplus", "mlp_softmax"]
kv_vals = [4, 8, 16, 32]

print("\n" + "=" * 65)
print("  Results table  |  seq=256  |  mean val acc over seed0 & seed1")
print("=" * 65)
header = f"{'mode':<18}" + "".join(f"kv={k:<8}" for k in kv_vals)
print(header)
print("-" * 65)

for mode in modes:
    row = f"{mode:<18}"
    for kv in kv_vals:
        val = summary.get((mode, kv, 256), None)
        cell = f"{val:.2f}%" if val is not None else "n/a"
        row += f"{cell:<12}"
    print(row)

print("=" * 65)

# ------- scaling story: kv=32, seq=512 -------
print("\n" + "=" * 50)
print("  Scaling check  |  kv=32, seq=512")
print("=" * 50)
for mode in modes:
    val = summary.get((mode, 32, 512), None)
    cell = f"{val:.2f}%" if val is not None else "n/a"
    print(f"  {mode:<18}  {cell}")
print("=" * 50)

# ------- save to csv -------
rows = []
for (mode, kv, seq), mean_acc in sorted(summary.items()):
    rows.append({"mode": mode, "kv": kv, "seq": seq, "mean_val_acc": mean_acc})

with open("results_table.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["mode", "kv", "seq", "mean_val_acc"])
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved full table to results_table.csv")
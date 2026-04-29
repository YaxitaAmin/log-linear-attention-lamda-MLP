# log-linear-attention-λ-MLP

Research project comparing **fixed-λ** vs **learned-λ** (MLP-softplus / MLP-softmax) variants of hierarchical linear attention on language modeling and synthetic tasks.

---

## Results Summary

| Hidden | Steps | Seed | Baseline (fixed-λ) | MLP Softmax | MLP Softplus |
|--------|-------|------|--------------------|-------------|--------------|
| 256    | 20k   | 0    | 328.19 ✓           | 328.15 ✓    | —            |
| 512    | 20k   | 0    | 276.20 ✓           | 277.84      | —            |
| 256    | 40k   | 0    | 256.62 ✓           | 257.65      | 257.03       |
| 256    | 40k   | 1    | 257.32             | 257.23      | 256.55 ✓     |
| 256    | 40k   | 2    | 257.13             | 257.06      | 256.89 ✓     |
| **512**| **40k**| **0**| **224.71**        | **226.06**  | **218.57 ✓ best** |

**MLP-Softplus achieves best overall PPL of 218.57** at hidden=512, 40k steps, outperforming the fixed-λ baseline.

---

## Setup

### Local (WSL / Linux)

```bash
git clone https://github.com/YaxitaAmin/log-linear-attention-lamda-MLP.git
cd log-linear-attention-lamda-MLP
python -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Docker

```bash
# Build
docker build -t hattention .

# Run interactively (GPU)
docker run --gpus all -it --rm \
  -v $(pwd)/logs:/workspace/log-linear-attention-lamda-MLP/logs \
  hattention

# Run inference directly from docker
docker run --gpus all --rm \
  -v $(pwd)/logs:/workspace/log-linear-attention-lamda-MLP/logs \
  hattention \
  python inference_lm.py \
    --checkpoint_dir logs/mlp_softplus_dim512_seed0 \
    --mode mlp_softplus --hidden_size 512 --num_heads 8
```

---

## Inference — Language Modeling (`inference_lm.py`)

Evaluates a saved checkpoint on the WikiText-103 validation (or test) set and prints perplexity.

### Single checkpoint

```bash
# Best result: MLP-Softplus, hidden=512 → expect PPL ≈ 218.57
python inference_lm.py \
  --checkpoint_dir logs/mlp_softplus_dim512_seed0 \
  --mode mlp_softplus \
  --hidden_size 512 \
  --num_heads 8

# Fixed-λ baseline → expect PPL ≈ 224.71
python inference_lm.py \
  --checkpoint_dir logs/fixed_dim512_seed0 \
  --mode fixed \
  --hidden_size 512 \
  --num_heads 8

# MLP-Softmax baseline → expect PPL ≈ 226.06
python inference_lm.py \
  --checkpoint_dir logs/mlp_softmax_dim512_seed0 \
  --mode mlp_softmax \
  --hidden_size 512 \
  --num_heads 8
```

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint_dir` | required | Path to folder containing `model.pt` / `best_model.pt` |
| `--mode` | required | `mlp_softplus` / `mlp_softmax` / `fixed` / `softmax` |
| `--hidden_size` | 256 | Must match training config |
| `--num_heads` | 4 | Must match training config (use `8` for dim-512 runs) |
| `--num_layers` | 6 | Must match training config |
| `--seq_len` | 512 | Sequence length for evaluation batches |
| `--batch_size` | 8 | Evaluation batch size |
| `--split` | `validation` | `validation` or `test` |
| `--ckpt_name` | `best_model.pt` | Filename of checkpoint (auto-detects `model.pt` too) |
| `--compare` | false | Run all runs in `COMPARE_RUNS` table and print summary |

### Compare all runs at once

Edit the `COMPARE_RUNS` list at the top of `inference_lm.py` to match your log directories, then:

```bash
python inference_lm.py --compare
```

Output example:
```
========================================================================
  DIR                                      MODE            H       PPL    STORED
────────────────────────────────────────────────────────────────────────
  mlp_softplus_dim512_seed0                mlp_softplus  512    218.57    218.57  ◀ BEST
  fixed_dim512_seed0                       fixed         512    224.71    224.71
  mlp_softmax_dim512_seed0                 mlp_softmax   512    226.06    226.06
========================================================================
```

Results are saved to `eval_comparison.json`.

---

## Inference — Synthetic Tasks (`inference.py`)

Evaluates checkpoints on **Selective Copy (scp)** and **MQAR** tasks.

### Single eval

```bash
# Selective copy, mlp_softplus, seq=256
python inference.py eval scp \
  --lambda_mode mlp_softplus \
  --train_seq_len 256 \
  --num_tokens 16 \
  --seed 0

# MQAR, fixed lambda, seq=128
python inference.py eval mqar \
  --lambda_mode fixed \
  --train_seq_len 128 \
  --num_kv_pairs 4 \
  --seed 0
```

### Compare all 3 modes side by side

```bash
python inference.py compare scp \
  --train_seq_len 256 \
  --num_tokens 16 \
  --seed 0
```

### Generalization test (train on short, eval on long)

```bash
python inference.py eval scp \
  --lambda_mode mlp_softplus \
  --train_seq_len 256 \
  --eval_seq_len 512 \
  --num_tokens 16 \
  --seed 0
```

### Sweep over seq_lens × seeds

```bash
python inference.py sweep_eval scp \
  --lambda_mode mlp_softplus \
  --seq_lens 128,256,512 \
  --seeds 0,1,2 \
  --eval_seq_len 512 \
  --num_tokens 16
```

---

## Training

```bash
# Language modeling (WikiText-103)
python train_lm.py \
  --mode mlp_softplus \
  --hidden_size 512 \
  --num_heads 8 \
  --max_steps 40000 \
  --output_dir logs/mlp_softplus_dim512_seed0

# Selective copy
python train_selective_copy.py --lambda_mode mlp_softplus --seq_len 256 --seed 0

# MQAR
python train_mqar.py --lambda_mode mlp_softplus --seq_len 128 --seed 0
```

---

## Checkpoint Structure

Each run folder contains:
```
logs/<run_name>/
├── model.pt          # saved weights
├── results.json      # training config + best val PPL
├── train_losses.json
└── val_losses.json
```

---

## Branch 612

This branch adds:
- `inference_lm.py` — full evaluation script for LM checkpoints with PPL reporting
- Updated `Dockerfile` — auto-installs hattention package, sets PYTHONPATH correctly
- This README

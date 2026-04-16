#!/bin/bash

# Language modeling experiment launcher
# Runs fixed, mlp_softplus, mlp_softmax variants with multiple seeds

set -e

# Configuration
LAMBDA_MODES=("fixed" "mlp_softplus" "mlp_softmax")
SEEDS=(0 1 2)
MLP_HIDDEN_DIMS=(64)
MODEL_DIM=256
SEQ_LEN=1024
BATCH_SIZE=8
MAX_STEPS=5000
LR=1e-4
OUTPUT_DIR="results/lm"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Language Modeling Experiments (Phase 2)"
echo "=========================================="
echo "Variants: ${LAMBDA_MODES[@]}"
echo "Seeds: ${SEEDS[@]}"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

TOTAL_RUNS=$((${#LAMBDA_MODES[@]} * ${#SEEDS[@]} * ${#MLP_HIDDEN_DIMS[@]}))
CURRENT_RUN=0

for mode in "${LAMBDA_MODES[@]}"; do
  for dh in "${MLP_HIDDEN_DIMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      CURRENT_RUN=$((CURRENT_RUN + 1))
      
      echo ""
      echo "[Run $CURRENT_RUN/$TOTAL_RUNS] lambda_mode=$mode mlp_hidden_dim=$dh seed=$seed"
      echo "---"
      
      python3 train_lm.py \
        --lambda_mode "$mode" \
        --mlp_hidden_dim "$dh" \
        --model_dim "$MODEL_DIM" \
        --seq_len "$SEQ_LEN" \
        --batch_size "$BATCH_SIZE" \
        --max_steps "$MAX_STEPS" \
        --lr "$LR" \
        --seed "$seed" \
        --output_dir "$OUTPUT_DIR"
    done
  done
done

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

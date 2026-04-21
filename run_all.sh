#!/bin/bash
echo "Starting mlp run..."
python3 train_lm.py --mode mlp --hidden_size 256 --num_layers 6 --num_heads 4 --seq_len 512 --batch_size 8 --max_steps 20000 --val_every 1000 --patience 10 --output_dir results/lm_mlp

echo "Starting fixed run..."
python3 train_lm.py --mode fixed --hidden_size 256 --num_layers 6 --num_heads 4 --seq_len 512 --batch_size 8 --max_steps 20000 --val_every 1000 --patience 10 --output_dir results/lm_fixed

echo "Starting softmax run..."
python3 train_lm.py --mode softmax --hidden_size 256 --num_layers 6 --num_heads 4 --seq_len 512 --batch_size 8 --max_steps 20000 --val_every 1000 --patience 10 --output_dir results/lm_softmax

echo "All done!"

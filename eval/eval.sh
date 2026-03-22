#!/usr/bin/env bash
# Train + evaluate EBLS on 8×H100 (or fewer GPUs)
set -euo pipefail

GPU_COUNT="${GPU_COUNT:-8}"
SEED="${SEED:-1337}"

RUN_ID="ebls_seed${SEED}" \
SEED="$SEED" \
DATA_PATH=./parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node="$GPU_COUNT" train_gpt.py

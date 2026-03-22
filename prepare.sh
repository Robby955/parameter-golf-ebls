#!/usr/bin/env bash
# One-command setup for EBLS training
set -euo pipefail

pip install -q sentencepiece huggingface_hub zstandard

if [ ! -d parameter-golf ]; then
    git clone https://github.com/openai/parameter-golf.git
fi

cd parameter-golf
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards "${TRAIN_SHARDS:-80}"
cd ..

echo "Setup complete. Run: bash eval/eval.sh"

#!/usr/bin/env bash
# Run all model experiments on FMA-Medium + Bollywood.
#
# Prerequisites:
#   1. fma_medium/ directory downloaded (scripts/download_fma_medium.sh)
#   2. bollywood/ directory populated (scripts/collect_bollywood.py)
#   3. conda environment 'torch' is active
#
# Usage:
#   conda activate torch
#   bash scripts/run_fma_medium_experiments.sh [--no_bollywood] [--zero_shot_only]

set -euo pipefail

INCLUDE_BOLLYWOOD=true
ZERO_SHOT_ONLY=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no_bollywood) INCLUDE_BOLLYWOOD=false; shift;;
    --zero_shot_only) ZERO_SHOT_ONLY=true; shift;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

BOLLYWOOD_FLAG=""
if $INCLUDE_BOLLYWOOD; then
  BOLLYWOOD_FLAG="--include_bollywood"
fi

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_DIR}"

echo "=============================="
echo " FMA-Medium Experiment Suite"
echo "=============================="
echo " Bollywood: ${INCLUDE_BOLLYWOOD}"
echo " Zero-shot only: ${ZERO_SHOT_ONLY}"
echo ""

# __ MERT-95M __
echo "--- MERT-95M zero-shot ---"
python models/mert/finetune.py --mode zero_shot --model_size 95m \
  --dataset medium ${BOLLYWOOD_FLAG}

if ! $ZERO_SHOT_ONLY; then
  echo "--- MERT-95M fine-tune ---"
  python models/mert/finetune.py --mode finetune --model_size 95m \
    --dataset medium ${BOLLYWOOD_FLAG} --epochs 15 --batch_size 8
fi

# __ MERT-330M __
echo "--- MERT-330M zero-shot ---"
python models/mert/finetune.py --mode zero_shot --model_size 330m \
  --dataset medium ${BOLLYWOOD_FLAG}

if ! $ZERO_SHOT_ONLY; then
  echo "--- MERT-330M fine-tune ---"
  python models/mert/finetune.py --mode finetune --model_size 330m \
    --dataset medium ${BOLLYWOOD_FLAG} --epochs 15 --batch_size 8
fi

# __ AST __
echo "--- AST zero-shot ---"
python models/ast/finetune.py --mode zero_shot \
  --dataset medium ${BOLLYWOOD_FLAG}

if ! $ZERO_SHOT_ONLY; then
  echo "--- AST fine-tune ---"
  python models/ast/finetune.py --mode finetune \
    --dataset medium ${BOLLYWOOD_FLAG} --epochs 20 --batch_size 16
fi

# __ CLAP-LAION __
echo "--- CLAP-LAION zero-shot ---"
python models/clap/finetune.py --mode zero_shot --variant laion \
  --dataset medium ${BOLLYWOOD_FLAG}

if ! $ZERO_SHOT_ONLY; then
  echo "--- CLAP-LAION fine-tune ---"
  python models/clap/finetune.py --mode finetune --variant laion \
    --dataset medium ${BOLLYWOOD_FLAG} --epochs 20 --batch_size 8
fi

# __ CLAP-Microsoft __
echo "--- CLAP-Microsoft zero-shot ---"
python models/clap/finetune_microsoft.py --mode zero_shot \
  --dataset medium ${BOLLYWOOD_FLAG}

if ! $ZERO_SHOT_ONLY; then
  echo "--- CLAP-Microsoft fine-tune ---"
  python models/clap/finetune_microsoft.py --mode finetune \
    --dataset medium ${BOLLYWOOD_FLAG} --epochs 20 --batch_size 8
fi

# __ MusicLDM-VAE __
echo "--- MusicLDM-VAE zero-shot ---"
python models/musicldm/finetune.py --mode zero_shot \
  --dataset medium ${BOLLYWOOD_FLAG} --save_embeddings

if ! $ZERO_SHOT_ONLY; then
  echo "--- MusicLDM-VAE fine-tune ---"
  python models/musicldm/finetune.py --mode finetune \
    --dataset medium ${BOLLYWOOD_FLAG} --epochs 20 --batch_size 16
fi

echo ""
echo "=== All FMA-Medium experiments complete ==="

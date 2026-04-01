#!/usr/bin/env bash
# Sequential fine-tuning on FMA-Medium (one model at a time → one GPU job at a time).
# Logs stdout/stderr to results/logs/ and per-epoch CSV + confusion matrices from each Python run.
#
# Prerequisites:
#   conda activate torch   # or your env with torch, transformers, diffusers, etc.
#   FMA_MEDIUM_AUDIO_DIR / fma_medium/ populated (see scripts/download_fma_medium.sh)
#
# Screen (recommended):
#   screen -S fma_ft
#   bash scripts/overnight_finetune_medium.sh
#   # Ctrl+A, D to detach
#
# Optional env:
#   CUDA_VISIBLE_DEVICES=0   # default: all visible GPUs; script still runs one process at a time
#
# GPU: use the same `python` as `conda activate torch` (run `which python`). If training is on CPU,
# your shell may be using a different interpreter without CUDA PyTorch — or add to each python line:
#   --device cuda
# which exits with an error if CUDA is not visible to PyTorch.
#   INCLUDE_BOLLYWOOD=1      # set to add --include_bollywood
#   MAX_EPOCHS=40            # upper cap per model; early stopping may finish sooner
#   EARLY_STOP=5             # epochs without val acc improvement before stop
#   SLEEP_SEC=120            # pause between models (lets GPU cool / clears pending work)

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_DIR}"

MAX_EPOCHS="${MAX_EPOCHS:-40}"
EARLY_STOP="${EARLY_STOP:-5}"
SLEEP_SEC="${SLEEP_SEC:-120}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${BASE_DIR}/results/logs/overnight_finetune_medium_${RUN_TS}.log"
mkdir -p "${BASE_DIR}/results/logs"

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=============================================="
echo " Overnight fine-tune | FMA-Medium | ${RUN_TS}"
echo " Master log: ${MASTER_LOG}"
echo " Max epochs/model: ${MAX_EPOCHS} | Early stop patience: ${EARLY_STOP}"
echo " Sleep between jobs: ${SLEEP_SEC}s"
echo "=============================================="

BOLLYWOOD_FLAG=()
if [[ "${INCLUDE_BOLLYWOOD:-0}" == "1" ]]; then
  BOLLYWOOD_FLAG=(--include_bollywood)
  echo "Including Bollywood tracks."
fi

# Shared flags: temporal sampling, early stopping, long enough max epochs
COMMON=(
  --mode finetune
  --dataset medium
  "${BOLLYWOOD_FLAG[@]}"
  --epochs "${MAX_EPOCHS}"
  --early_stop_patience "${EARLY_STOP}"
  --early_stop_min_delta 0.0001
)

cooldown() {
  echo "--- Cooldown ${SLEEP_SEC}s ($(date -Iseconds)) ---"
  sleep "${SLEEP_SEC}"
}

echo ">>> AST"
python models/ast/finetune.py "${COMMON[@]}" \
  --batch_size 16 --grad_accum 2 --lr_backbone 1e-5 --lr_head 1e-3
cooldown

echo ">>> MERT-95M"
python models/mert/finetune.py "${COMMON[@]}" \
  --model_size 95m --batch_size 8 --grad_accum 4 --lr_backbone 5e-5 --lr_head 1e-3
cooldown

echo ">>> MERT-330M"
python models/mert/finetune.py "${COMMON[@]}" \
  --model_size 330m --batch_size 8 --grad_accum 4 --lr_backbone 5e-5 --lr_head 1e-3
cooldown

echo ">>> CLAP-Laion"
python models/clap/finetune.py "${COMMON[@]}" \
  --variant laion --batch_size 8 --grad_accum 2 --lr_audio 5e-5 --lr_head 1e-3
cooldown

echo ">>> CLAP-Microsoft"
python models/clap/finetune.py "${COMMON[@]}" \
  --variant microsoft --batch_size 8 --grad_accum 2 --lr_audio 5e-5 --lr_head 1e-3
cooldown

echo ">>> MusicLDM-VAE"
python models/musicldm/finetune.py "${COMMON[@]}" \
  --batch_size 16 --grad_accum 2 --lr_backbone 5e-5 --lr_head 1e-3

echo ""
echo "=== Overnight fine-tune suite finished at $(date -Iseconds) ==="
echo "Per-epoch CSVs: ${BASE_DIR}/results/logs/finetune/"
echo "Confusion matrices + curves: ${BASE_DIR}/results/figures/confmat_* ${BASE_DIR}/results/figures/*training_curves*"
echo "Run JSON metadata: ${BASE_DIR}/results/runs/"

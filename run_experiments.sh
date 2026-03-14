#!/usr/bin/env bash
# run_experiments.sh
# Run all zero-shot evaluations, then fine-tune AST and evaluate it.
# Usage: bash run_experiments.sh
# Make sure your conda environment is active before running:
#   conda activate commmon

set -e
cd "$(dirname "$0")"

echo "================================================================"
echo " CS5100-FAI — Experiment Runner"
echo " $(date)"
echo "================================================================"

# __ Step 1: Zero-shot linear-probe evaluation for all 4 models __
echo ""
echo "[1/3] Zero-shot evaluation: MERT-95M, MERT-330M, CLAP-LAION, AST"
echo "      (frozen encoder → logistic regression, ~30–90 min per model on GPU)"
echo ""

python evaluate.py \
    --model mert-95m \
    --model mert-330m \
    --model clap-laion \
    --model ast

echo ""
echo "[1/3] Done. Results saved to results/runs/ and results/figures/"

# __ Step 2: Fine-tune AST end-to-end __
echo ""
echo "[2/3] Fine-tuning AST (end-to-end, 20 epochs)"
echo "      Estimated time: ~1–2 hours on GPU"
echo ""

python models/ast/finetune.py \
    --mode finetune \
    --epochs 20 \
    --batch_size 8 \
    --grad_accum 4 \
    --lr_backbone 1e-5 \
    --lr_head 1e-3

echo ""
echo "[2/3] Done. Checkpoint saved to results/checkpoints/ast/"

# __ Step 3: Zero-shot run for AST (skip if already done in step 1) __
# This is included so you can run AST zero-shot independently if needed.
# echo ""
# echo "[3/3] AST zero-shot only (optional standalone)"
# python models/ast/finetune.py --mode zero_shot

echo ""
echo "================================================================"
echo " All experiments complete!"
echo " Open notebooks/progress_report.ipynb and run all cells"
echo " to see the updated comparison table and figures."
echo "================================================================"

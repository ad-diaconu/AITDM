#!/usr/bin/env bash
#SBATCH --job-name=wisdm-central
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-9%2
#SBATCH --output=logs/%x-%A_%a.out
# ---------------------------------------------------------------------------
# Centralised WISDM baseline: 2 models x 5 subject-wise folds = 10 array tasks,
# each running seeds 1 2 3 on one GPU. Submit from the repository root:
#
#   sbatch wisdm_baseline/fep/train_array.sbatch
#   sbatch --array=3 wisdm_baseline/fep/train_array.sbatch     # re-run one task
#
# Task id t -> model = (fcn, lstm)[t / 5], fold = t % 5. Finished seeds are skipped.
# The student account may use dgxa100 / dgxh100 / ucsx (not ml) and at most 2 jobs
# at a time (QoS normal), hence %2. One A100: ~1 s per epoch, a few minutes per task.
# ---------------------------------------------------------------------------
set -euo pipefail

IMAGE="${IMAGE:-$HOME/NNProj/convnext.sif}"   # torch 2.9.1 + CUDA 12.6 + numpy
MODELS=(fcn lstm)
TASK=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODELS[$((TASK / 5))]}
FOLD=$((TASK % 5))

echo "=== job ${SLURM_ARRAY_JOB_ID:-?}_${TASK} on $(hostname), $(date)"
echo "    model=$MODEL fold=$FOLD image=$IMAGE GPUs=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

apptainer exec --nv "$IMAGE" python -u wisdm_baseline/train.py \
    --data data/wisdm/windows.npz --model "$MODEL" --fold "$FOLD" \
    --seeds 1 2 3 --epochs 30 --patience 8 --batch-size 64 --lr 1e-3 \
    --out outputs/wisdm "$@"

echo "=== done $(date)"

#!/usr/bin/env bash
#SBATCH --job-name=aitdm-centralised_array
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-9%2
# ---------------------------------------------------------------------------
# Centralised WISDM baseline (experiments/centralised.py): 2 models x 5 subject-wise folds
# = 10 array tasks, each running seeds 1 2 3 on one GPU.
#
#   bash Fep/submit.sh centralised_array
#   bash Fep/submit.sh --array=3 centralised_array                # re-run one task
#   AITDM_IMAGE=~/NNProj/convnext.sif bash Fep/submit.sh centralised_array   # as the results were made
#
# Task id t -> model = (fcn, lstm)[t / 5], fold = t % 5. Finished seeds are skipped.
# Extra arguments go to centralised.py. The student account may use dgxa100 / dgxh100 / ucsx
# (not ml) and at most 2 jobs at a time (QoS normal), hence %2 (bash Fep/my_partitions.sh).
# One A100: ~1 s per epoch, a few minutes per task.
# ---------------------------------------------------------------------------
source Fep/jobs/_job_common.sh centralised.py "$@"

MODELS=(fcn lstm)
TASK=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODELS[$((TASK / 5))]}
FOLD=$((TASK % 5))
echo "    model=$MODEL fold=$FOLD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"${PY[@]}" -u experiments/centralised.py \
    --data data/wisdm/windows.npz --model "$MODEL" --fold "$FOLD" \
    --seeds 1 2 3 --epochs 30 --patience 8 --batch-size 64 --lr 1e-3 \
    --out outputs/wisdm "$@"

echo "=== done $(date)"

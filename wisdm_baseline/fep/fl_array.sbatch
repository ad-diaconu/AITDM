#!/usr/bin/env bash
#SBATCH --job-name=wisdm-fl
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
# ---------------------------------------------------------------------------
# Federated WISDM experiments: one array task = one line of a jobs file
# (run_fl.py arguments: split, clients, model, strategy, ...). Inside the task the
# 5 folds x 3 seeds run one after the other; with --gres=gpu:N the folds are
# spread over N GPUs (one run_fl.py process per GPU). Submit from the repository root:
#
#   sbatch --array=0-8%2 wisdm_baseline/fep/fl_array.sbatch wisdm_baseline/fep/fl_primary.txt
#   sbatch --array=4 --gres=gpu:2 --cpus-per-task=20 wisdm_baseline/fep/fl_array.sbatch <jobs file>
#
# Finished (fold, seed) runs are skipped, so failed tasks can simply be resubmitted.
# The student account may use dgxa100 / dgxh100 / ucsx and run at most 2 jobs at a time.
# ---------------------------------------------------------------------------
set -euo pipefail

JOBS_FILE=${1:?usage: sbatch [--array=...] fl_array.sbatch <jobs file>}
IMAGE="${IMAGE:-$HOME/NNProj/convnext.sif}"           # torch 2.9.1 + CUDA 12.6
PYTHON="${PYTHON:-$HOME/.venvs/aitdm-fl/bin/python}"  # + flwr[simulation] 1.41.0
TASK=${SLURM_ARRAY_TASK_ID:-0}
ARGS=$(grep -v -E '^\s*(#|$)' "$JOBS_FILE" | sed -n "$((TASK + 1))p")
[ -n "$ARGS" ] || { echo "no line $TASK in $JOBS_FILE" >&2; exit 1; }

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
NGPU=${#GPUS[@]}
CPUS_PER_PROC=$(( ${SLURM_CPUS_PER_TASK:-10} / NGPU ))
echo "=== job ${SLURM_ARRAY_JOB_ID:-?}_${TASK} on $(hostname), $(date)"
echo "    args: $ARGS | GPUs: ${CUDA_VISIBLE_DEVICES:-none} | $CPUS_PER_PROC CPUs per process"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader || true

FOLDS=(0 1 2 3 4)
PIDS=()
for g in $(seq 0 $((NGPU - 1))); do
    MINE=()
    for f in "${FOLDS[@]}"; do [ $((f % NGPU)) -eq "$g" ] && MINE+=("$f"); done
    [ ${#MINE[@]} -gt 0 ] || continue
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=${GPUS[$g]} apptainer exec --nv "$IMAGE" "$PYTHON" -u wisdm_baseline/run_fl.py \
        $ARGS --folds "${MINE[@]}" --seeds 1 2 3 --gpus 1 --cpus "$CPUS_PER_PROC" &
    PIDS+=($!)
done
STATUS=0
for p in "${PIDS[@]}"; do wait "$p" || STATUS=1; done
echo "=== done $(date), status $STATUS"
exit $STATUS

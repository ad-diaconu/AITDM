#!/usr/bin/env bash
#SBATCH --job-name=aitdm-fl_array
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=04:00:00
# ---------------------------------------------------------------------------
# Federated WISDM experiments: one array task = one line of a run list in experiments/runs/
# (experiments/run_seeds.py arguments: split, clients, model, strategy, ...). Inside the task
# the 5 folds x 3 seeds run one after the other; with --gres=gpu:N the folds are spread over
# N GPUs (one run_seeds.py process per GPU).
#
#   bash Fep/submit.sh --array=0-8%2 fl_array experiments/runs/fl_primary.txt
#   bash Fep/submit.sh --array=4 --gres=gpu:2 --cpus-per-task=20 --mem=64G fl_array experiments/runs/fl_secondary.txt
#   AITDM_IMAGE=~/NNProj/convnext.sif bash Fep/submit.sh --array=0-8%2 fl_array <run list>   # as the results were made
#
# Finished (fold, seed) runs are skipped, so failed tasks can simply be resubmitted.
# The student account may use dgxa100 / dgxh100 / ucsx and run at most 2 jobs at a time.
# ---------------------------------------------------------------------------
source Fep/jobs/_job_common.sh run_seeds.py "$@"

JOBS_FILE=${1:?usage: bash Fep/submit.sh --array=... fl_array <run list>}
TASK=${SLURM_ARRAY_TASK_ID:-0}
ARGS=$(grep -v -E '^\s*(#|$)' "$JOBS_FILE" | sed -n "$((TASK + 1))p")
[ -n "$ARGS" ] || { echo "no line $TASK in $JOBS_FILE" >&2; exit 1; }
# a line is parsed like a shell command line, so quoted --config "k='v' ..." values work
eval "ARGV=($ARGS)"

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
NGPU=${#GPUS[@]}
CPUS_PER_PROC=$(( ${SLURM_CPUS_PER_TASK:-10} / NGPU ))
echo "    args: $ARGS | GPUs: ${CUDA_VISIBLE_DEVICES:-none} | $CPUS_PER_PROC CPUs per process"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader || true

FOLDS=(0 1 2 3 4)
PIDS=()
for g in $(seq 0 $((NGPU - 1))); do
    MINE=()
    for f in "${FOLDS[@]}"; do [ $((f % NGPU)) -eq "$g" ] && MINE+=("$f"); done
    [ ${#MINE[@]} -gt 0 ] || continue
    CUDA_VISIBLE_DEVICES=${GPUS[$g]} "${PY[@]}" -u experiments/run_seeds.py \
        "${ARGV[@]}" --folds "${MINE[@]}" --seeds 1 2 3 --gpus 1 --cpus "$CPUS_PER_PROC" &
    PIDS+=($!)
done
STATUS=0
for p in "${PIDS[@]}"; do wait "$p" || STATUS=1; done
echo "=== done $(date), status $STATUS"
exit $STATUS

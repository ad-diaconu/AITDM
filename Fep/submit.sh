#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Submit one of the jobs in Fep/jobs/ to Slurm, from anywhere in the clone.
# The job runs in the repository root and its log goes to
# ~/AITDM_Project/slurm/<job>-<jobid>.out.
#
#   bash Fep/submit.sh [SBATCH_OPTIONS...] JOB [JOB_ARGUMENTS...]
#
# Examples:
#   bash Fep/submit.sh smoke_test
#   bash Fep/submit.sh --array=0-8%2 fl_array experiments/runs/fl_primary.txt
#   bash Fep/submit.sh centralised_array
#   bash Fep/submit.sh --gres=gpu:1 simulation data/partitions/wisdm/s2_K5/fold0 "num-server-rounds=10"
#   bash Fep/submit.sh command python experiments/summarize_fl.py
#
# Jobs: the file names in Fep/jobs/ without .sbatch.
# Any sbatch option placed before JOB overrides the #SBATCH lines of the job.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AITDM_PROJECT="${AITDM_PROJECT:-$HOME/AITDM_Project}"
JOBS_DIR="$REPO_ROOT/Fep/jobs"

SBATCH_OPTS=()
while [ $# -gt 0 ] && [[ "$1" == -* ]]; do
    case "$1" in
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        # Options with a separate value (-p ml, -t 01:00:00, ...)
        -p|-t|-c|-n|-N|-A|-q|-w|-x|-J|--mem|--gres|--time|--partition|--account|--qos|--nodes|--cpus-per-task)
            SBATCH_OPTS+=("$1" "$2"); shift 2 ;;
        *) SBATCH_OPTS+=("$1"); shift ;;
    esac
done
if [ $# -lt 1 ]; then
    echo "Usage: bash Fep/submit.sh [SBATCH_OPTIONS...] JOB [JOB_ARGUMENTS...]" >&2
    echo "Jobs: $(cd "$JOBS_DIR" && ls ./*.sbatch | sed 's|^\./||; s|\.sbatch$||' | tr '\n' ' ')" >&2
    exit 1
fi
JOB="$1"
shift
JOB_FILE="$JOBS_DIR/$JOB.sbatch"
if [ ! -f "$JOB_FILE" ]; then
    echo "ERROR: no job '$JOB' in Fep/jobs/" >&2
    exit 1
fi
if [ -z "${AITDM_IMAGE:-}" ] && [ ! -f "$REPO_ROOT/.venv/bin/activate" ]; then
    echo "ERROR: no .venv yet. Run first: bash Fep/setup_fep.sh (or set AITDM_IMAGE, see Fep/README.md)" >&2
    exit 1
fi

mkdir -p "$AITDM_PROJECT/slurm"
# Array jobs (--array=... here or in the job file): one log per task
LOG="%x-%j.out"
SHOWN="${JOB}-JOBID.out"
if [[ " ${SBATCH_OPTS[*]:-} " == *" --array"* ]] || grep -q '^#SBATCH --array' "$JOB_FILE"; then
    LOG="%x-%A_%a.out"
fi
JOB_ID=$(sbatch --parsable --chdir="$REPO_ROOT" \
    --output="$AITDM_PROJECT/slurm/$LOG" \
    "${SBATCH_OPTS[@]}" "$JOB_FILE" "$@")
JOB_ID="${JOB_ID%%;*}"
JOB_NAME=$(squeue -h -j "$JOB_ID" -o %j 2>/dev/null | head -1 || true)
SHOWN="${JOB_NAME:-$JOB}-$JOB_ID.out"
[ "$LOG" = "%x-%j.out" ] || SHOWN="${JOB_NAME:-$JOB}-${JOB_ID}_<task>.out"
echo "Submitted job $JOB_ID"
echo "Log:    $AITDM_PROJECT/slurm/$SHOWN"
echo "Follow: tail -f $AITDM_PROJECT/slurm/$SHOWN     Cancel: scancel $JOB_ID     Queue: squeue --me"

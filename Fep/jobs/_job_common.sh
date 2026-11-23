#!/usr/bin/env bash
# Sourced at the start of every job in Fep/jobs/ (never run directly).
# Slurm copies the job script elsewhere, so we find the repository from the
# working directory: Fep/submit.sh starts the job in the repository root.
#
# Python environment, two options:
#   * default: the repository's .venv (bash Fep/setup_fep.sh), as in the tutorial
#   * the container every result in outputs/ was produced with: set
#       AITDM_IMAGE=~/NNProj/convnext.sif            (torch 2.9.1 + CUDA 12.6)
#       AITDM_PYTHON=~/.venvs/aitdm-fl/bin/python    (+ flwr[simulation] 1.41.0, default)
#     e.g. AITDM_IMAGE=~/NNProj/convnext.sif bash Fep/submit.sh fl_array ...
# Jobs run Python as "${PY[@]}" so that both work.
set -euo pipefail
if [ ! -f scripts/_common.sh ]; then
    echo "ERROR: the job must start in the repository root." >&2
    echo "Submit it with: bash Fep/submit.sh <job> ... (or sbatch from the root)" >&2
    exit 1
fi
echo "=== job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname), $(date)"
echo "    partition ${SLURM_JOB_PARTITION:-?}, ${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-?}} CPUs, GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "    data -> $(readlink -f data 2>/dev/null || echo 'data/ (not linked: run Fep/setup_fep.sh)')"
echo "    command: $*"
# Provenance: a clone synced from a commit has a COMMIT file (short hash, "-dirty" for uncommitted code);
# centralised.py and run_seeds.py copy it into every run folder
if [ -f COMMIT ]; then
    export AITDM_COMMIT="$(cat COMMIT)"
elif git rev-parse --short HEAD >/dev/null 2>&1; then
    AITDM_COMMIT="$(git rev-parse --short HEAD)"
    [ -z "$(git status --porcelain -- data_splitting experiments fl_app 2>/dev/null)" ] || AITDM_COMMIT+="-dirty"
    export AITDM_COMMIT
fi
echo "    commit: ${AITDM_COMMIT:-unknown}"
if [ -n "${AITDM_IMAGE:-}" ]; then
    PY=(apptainer exec --nv "$AITDM_IMAGE" "${AITDM_PYTHON:-$HOME/.venvs/aitdm-fl/bin/python}")
    echo "    python: ${PY[*]}"
    export VIRTUAL_ENV="${VIRTUAL_ENV:-container}"  # scripts/_common.sh: do not look for .venv
else
    PY=(python)
fi
# shellcheck disable=SC1091
source scripts/_common.sh

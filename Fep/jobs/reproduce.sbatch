#!/usr/bin/env bash
#SBATCH --job-name=aitdm-reproduce
#SBATCH --partition=haswell
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
# ---------------------------------------------------------------------------
# experiments/reproduce.sh as one job (sequential, CPU). For the training steps use the GPU array
# jobs centralised_array and fl_array (parallel); use this for data and summary.
# Everything here on a GPU: bash Fep/submit.sh -p dgxa100 --gres=gpu:1 -t 12:00:00 reproduce
#   bash Fep/submit.sh reproduce data        # WISDM archive -> windows -> all partitions
#   bash Fep/submit.sh reproduce summary     # summaries and reports
#   bash Fep/submit.sh reproduce             # everything; resubmit if it hits the time limit
# ---------------------------------------------------------------------------
source Fep/jobs/_job_common.sh reproduce.sh "$@"
PY="${PY[*]}" bash experiments/reproduce.sh "$@"

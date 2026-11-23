#!/usr/bin/env bash
#SBATCH --job-name=aitdm-command
#SBATCH --partition=haswell
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
# ---------------------------------------------------------------------------
# Any other command of the tutorials, run on a compute node with .venv active.
#   bash Fep/submit.sh command python data_splitting/split_data.py --splits s1 s2 --clients 5
#   bash Fep/submit.sh command python experiments/summarize_fl.py
#   bash Fep/submit.sh command python explainability/error_analysis.py --run outputs/wisdm_fl/<exp>/<method>/fold0/seed1
# ---------------------------------------------------------------------------
source Fep/jobs/_job_common.sh "$@"
"$@"

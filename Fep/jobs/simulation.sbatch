#!/usr/bin/env bash
#SBATCH --job-name=aitdm-simulation
#SBATCH --partition=haswell
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
# ---------------------------------------------------------------------------
# One federated run (server + all clients) with Flower's Simulation Runtime:
# the same as scripts/run_simulation.sh on your laptop, same arguments.
#   bash Fep/submit.sh simulation data/partitions/wisdm/s2_K5/fold0
#   bash Fep/submit.sh simulation data/partitions/wisdm/s2_K5/fold0 "num-server-rounds=10 strategy='fedprox'"
#   bash Fep/submit.sh -p ucsx --gres=gpu:1 simulation data/partitions/wisdm/s2_K5/fold0
# Results: outputs/sim-<split>_K<K>-fold<f>-<timestamp>-job<id>/ (git-ignored: experiment results
# come from fl_array / experiments/run_seeds.py).
# ---------------------------------------------------------------------------
source Fep/jobs/_job_common.sh run_simulation.sh "$@"
bash scripts/run_simulation.sh "$@"

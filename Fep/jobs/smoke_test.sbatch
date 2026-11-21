#!/usr/bin/env bash
#SBATCH --job-name=aitdm-smoke_test
#SBATCH --partition=haswell
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
# ---------------------------------------------------------------------------
# End-to-end check on synthetic data (no download): split, 1 federated round,
# explanations, centralised baseline. About 5 minutes.
#   bash Fep/submit.sh smoke_test
#   bash Fep/submit.sh -p ucsx --gres=gpu:1 smoke_test     # on a GPU
# ---------------------------------------------------------------------------
source Fep/jobs/_job_common.sh smoke_test
"${PY[@]}" -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
# container mode: no `flwr` command line tool outside the .venv, so skip the `flwr run` step
if [ -n "${AITDM_IMAGE:-}" ]; then export SMOKE_SKIP_FLWR_RUN=1; fi
PY="${PY[*]}" bash tests/smoke_test.sh

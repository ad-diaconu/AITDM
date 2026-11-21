#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run the whole federation (server + all clients) on this machine with
# Flower's Simulation Runtime – ONE command, ONE terminal.
#
# Usage:
#   bash scripts/run_simulation.sh [PARTITIONS_DIR] [RUN_CONFIG_OVERRIDES]
#
# Examples:
#   bash scripts/run_simulation.sh
#   bash scripts/run_simulation.sh data/partitions/wisdm/s0_K5/fold0
#   bash scripts/run_simulation.sh data/partitions/wisdm/s2_K5/fold0 "num-server-rounds=10 strategy='fedprox'"
#
# The number of simulated clients is read from <PARTITIONS_DIR>/meta.json.
# Results (results.json, final_model.pt) go to outputs/sim-<split>_K<K>-fold<f>-<timestamp>/
# (git-ignored: experiment results come from experiments/run_seeds.py).
# ---------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PARTITIONS="$(abspath "${1:-$DEFAULT_PARTITIONS}")"
EXTRA_CONFIG="${2:-}"
check_partitions "$PARTITIONS"
NUM_CLIENTS="$(num_clients_of "$PARTITIONS")"
# Results folder (SAVE_DIR=... to choose it, as experiments/run_seeds.py does)
# (Slurm jobs add their job id: two jobs may start in the same second)
SAVE_DIR="$(abspath "${SAVE_DIR:-$REPO_ROOT/outputs/sim-$(basename "$(dirname "$PARTITIONS")")-$(basename "$PARTITIONS")-$(date +%Y%m%d-%H%M%S)${SLURM_JOB_ID:+-job$SLURM_JOB_ID}}")"

# Resources Ray may use for the simulated clients. Inside a Slurm job they come
# from the allocation (Ray would otherwise see all CPUs of the node); set
# SIM_CPUS / SIM_GPUS / CLIENT_CPUS / CLIENT_GPUS to choose them yourself.
FEDERATION_CONFIG="num-supernodes=$NUM_CLIENTS"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SIM_CPUS="${SIM_CPUS:-${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}}"
    if [ -z "${SIM_GPUS:-}" ]; then
        SIM_GPUS=0
        if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]; then
            SIM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
        fi
    fi
fi
[ -n "${SIM_CPUS:-}" ] && FEDERATION_CONFIG+=" init-args-num-cpus=$SIM_CPUS"
[ -n "${SIM_GPUS:-}" ] && FEDERATION_CONFIG+=" init-args-num-gpus=$SIM_GPUS"
if [ -z "${CLIENT_GPUS:-}" ] && [ "${SIM_GPUS:-0}" -gt 0 ]; then
    # Share the GPUs between all clients so they can train at the same time
    CLIENT_GPUS=$(python -c "print(min(1.0, $SIM_GPUS / $NUM_CLIENTS))")
fi
[ -n "${CLIENT_CPUS:-}" ] && FEDERATION_CONFIG+=" client-resources-num-cpus=$CLIENT_CPUS"
[ -n "${CLIENT_GPUS:-}" ] && FEDERATION_CONFIG+=" client-resources-num-gpus=$CLIENT_GPUS"

# `flwr run .` starts a background "managed" SuperLink and keeps it alive.
# Restart it so that it uses THIS virtual environment and settings.
# Only ours: the pattern includes our port (Slurm jobs each have their own).
SIM_PORT="${FLWR_LOCAL_SUPERLINK_HTTP_API_PORT:-39091}"
SIM_PATTERN="[f]lower-superlink.*--simulation.* --port $SIM_PORT( |$)"
if pkill -u "$(id -u)" -f "$SIM_PATTERN" 2>/dev/null; then
    echo ">>> Restarting Flower's local simulation SuperLink"
    sleep 2
fi
if [ -n "${SLURM_JOB_ID:-}" ]; then
    # In a batch job nobody re-uses it afterwards: stop it when we are done
    trap 'pkill -u "$(id -u)" -f "$SIM_PATTERN" 2>/dev/null || true' EXIT
fi

echo ">>> Simulating $NUM_CLIENTS clients on $PARTITIONS ($FEDERATION_CONFIG)"
cd "$APP_DIR"
flwr run . --stream \
    --federation-config "$FEDERATION_CONFIG" \
    --run-config "partitions-dir='$PARTITIONS' save-dir='$SAVE_DIR' min-nodes=$NUM_CLIENTS $EXTRA_CONFIG"

# `flwr run` does not fail when the ServerApp crashes: check that results exist
if [ ! -f "$SAVE_DIR/results.json" ]; then
    echo ">>> The run FAILED - read the error (Traceback) above." >&2
    exit 1
fi
echo
echo ">>> Done. Model and metrics: $SAVE_DIR"

#!/usr/bin/env bash
# Shared settings for the run scripts. Sourced, never executed directly.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$REPO_ROOT/fl_app"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
# One fold of one split (data_splitting/split_data.py): data/partitions/wisdm/<split>_K<K>/fold<f>
DEFAULT_PARTITIONS="$REPO_ROOT/data/partitions/wisdm/s1_K5/fold0"

# Inside a Slurm job (e.g. on fep, see Fep/02_running_on_slurm.md) several jobs
# can share one node, but Flower always uses the same ports (simulation: 39091,
# 39093; deployment: 8000, 9094, ...) and keeps its state in ~/.flwr. A second
# job would then send its run to the first job's SuperLink. So every job gets
# its own block of 20 ports, computed from the job id (every process of the
# job finds the same ones), and its own Flower and Ray folders on the node.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    JOB_PORT_BASE=$((20000 + (SLURM_JOB_ID % 2000) * 20))
    export FLWR_LOCAL_SUPERLINK_HTTP_API_PORT="${FLWR_LOCAL_SUPERLINK_HTTP_API_PORT:-$JOB_PORT_BASE}"
    export FLWR_LOCAL_CONTROL_API_PORT="${FLWR_LOCAL_CONTROL_API_PORT:-$((JOB_PORT_BASE + 1))}"
    SUPERLINK_PORT="${SUPERLINK_PORT:-$((JOB_PORT_BASE + 2))}"
    SUPERNODE_PORT_BASE="${SUPERNODE_PORT_BASE:-$((JOB_PORT_BASE + 3))}"
    export FLWR_HOME="${FLWR_HOME:-/tmp/flwr-$USER-$SLURM_JOB_ID}"
    # Short path: Ray's socket paths must stay under 107 characters
    export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-$SLURM_JOB_ID}"
    mkdir -p "$FLWR_HOME" "$RAY_TMPDIR"
fi
SUPERLINK_PORT="${SUPERLINK_PORT:-8000}"
SUPERNODE_PORT_BASE="${SUPERNODE_PORT_BASE:-9094}"

# Name of the SuperLink connection used for deployment runs (~/.flwr/config.toml)
# (Fep/jobs/deployment.sbatch sets the address to the SuperLink of its job)
CONNECTION_NAME="${CONNECTION_NAME:-local-deployment}"
SUPERLINK_ADDRESS="${SUPERLINK_ADDRESS:-127.0.0.1:$SUPERLINK_PORT}"

# Activate the virtual environment (created by scripts/setup_env.sh; bin/ on Linux / macOS /
# WSL, Scripts/ for a Windows venv used from Git Bash)
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -f "$VENV_DIR/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
    elif [ -f "$VENV_DIR/Scripts/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/Scripts/activate"
    else
        echo "ERROR: no virtual environment in $VENV_DIR. Run: bash scripts/setup_env.sh" >&2
        exit 1
    fi
fi

# Everything is already installed in the venv: stop Flower from building a
# separate environment (and downloading PyTorch again) for every run.
export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1
# UTF-8 I/O everywhere (Flower's log stream prints box-drawing characters; matters on Windows)
export PYTHONUTF8=1
# Silence a harmless Ray warning printed in simulation mode
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Turn a (possibly relative) path into an absolute one. Flower runs the
# apps from its own working directory, so relative paths would break.
abspath() {
    if command -v cygpath >/dev/null 2>&1; then
        cygpath -am "$1"  # Git Bash on Windows: C:/... (Python cannot read /tmp/... inside run-config strings)
    else
        python -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$1"
    fi
}

# Fail early with a helpful message if the partitions were not created yet
check_partitions() {
    local dir="$1"
    if [ ! -f "$dir/test.npz" ] || [ ! -f "$dir/meta.json" ]; then
        echo "ERROR: no partitions found in $dir" >&2
        echo "Create them first, e.g.:" >&2
        echo "    python data_splitting/prepare_wisdm.py --input data/raw/wisdm/WISDM_at_v2.0_raw.txt" >&2
        echo "    python data_splitting/split_data.py --splits s1 --clients 5" >&2
        exit 1
    fi
}

# Number of clients stored in meta.json by split_data.py
num_clients_of() {
    python -c 'import json, sys; print(json.load(open(sys.argv[1]))["num_clients"])' "$1/meta.json"
}

# Add the deployment connection to ~/.flwr/config.toml (only once)
ensure_connection() {
    local config_file
    config_file="$(flwr config list 2>/dev/null | sed -n 's/^Flower Config file: //p')"
    config_file="${config_file:-$HOME/.flwr/config.toml}"
    if ! grep -q "^\[superlink\.$CONNECTION_NAME\]" "$config_file" 2>/dev/null; then
        echo ">>> Adding connection '$CONNECTION_NAME' ($SUPERLINK_ADDRESS) to $config_file"
        mkdir -p "$(dirname "$config_file")"
        printf '\n[superlink.%s]\naddress = "%s"\ninsecure = true\n' \
            "$CONNECTION_NAME" "$SUPERLINK_ADDRESS" >> "$config_file"
    fi
}

#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Creates a Python virtual environment in ./.venv and installs everything
# from requirements.txt (Flower, PyTorch, NumPy / pandas / SciPy, ...) plus the Flower app in
# fl_app/ (editable mode).
#
# Usage (from the repository root):
#   bash scripts/setup_env.sh              # default PyTorch build (CUDA on Linux)
#   bash scripts/setup_env.sh --cpu        # CPU-only PyTorch (much smaller download)
#   PYTHON=python3.12 bash scripts/setup_env.sh   # choose the interpreter
# If python3 is too old (e.g. Ubuntu 20.04 / 22.04 in WSL), python3.11-3.13 or a
# Python installed by uv is used instead (e.g. uv: curl -LsSf https://astral.sh/uv/install.sh | sh).
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
PYTHON="${PYTHON:-python3}"
TORCH_INDEX=""

for arg in "$@"; do
    case "$arg" in
        --cpu) TORCH_INDEX="https://download.pytorch.org/whl/cpu" ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

# --- 1. Check the Python version (Flower needs >= 3.11) --------------------
supported() {  # supported PYTHON : true if it exists and is 3.11 - 3.13
    command -v "$1" >/dev/null 2>&1 &&
        "$1" -c 'import sys; sys.exit(not ((3, 11) <= sys.version_info[:2] <= (3, 13)))' 2>/dev/null
}
# Old distributions (e.g. Ubuntu 20.04 / 22.04, common in WSL) ship an older
# python3. If PYTHON was not chosen explicitly, look for a supported one:
# python3.13 / 3.12 / 3.11 on the PATH, else a Python managed by uv (no sudo
# needed).
if [ "$PYTHON" = python3 ] && ! supported python3; then
    for candidate in python3.13 python3.12 python3.11; do
        if supported "$candidate"; then
            PYTHON="$candidate"
            break
        fi
    done
    if [ "$PYTHON" = python3 ] && command -v uv >/dev/null 2>&1; then
        echo ">>> python3 is too old; getting Python 3.12 from uv"
        uv python install 3.12
        PYTHON="$(uv python find --system 3.12)"  # --system: never an existing .venv
    fi
    if [ "$PYTHON" = python3 ]; then
        cat >&2 <<'EOF'
ERROR: Python 3.11-3.13 required, but python3 is older and no python3.11-3.13 was found.
Install one without sudo with uv, then re-run this script:
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.local/bin/env      # or open a new terminal
    bash scripts/setup_env.sh --cpu
(or with sudo on Ubuntu: sudo add-apt-repository ppa:deadsnakes/ppa &&
 sudo apt install python3.12 python3.12-venv, then PYTHON=python3.12 bash scripts/setup_env.sh)
EOF
        exit 1
    fi
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: '$PYTHON' not found. Install Python 3.11+ or set PYTHON=/path/to/python" >&2
    exit 1
fi
"$PYTHON" - <<'EOF'
import sys
if not ((3, 11) <= sys.version_info[:2] <= (3, 13)):
    sys.exit(f"ERROR: Python 3.11-3.13 required, found {sys.version.split()[0]}. "
             "Set PYTHON=python3.12 (or similar) and re-run.")
print(f"Using Python {sys.version.split()[0]}")
EOF

# --- 2. Create the virtual environment ------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo ">>> Creating virtual environment in $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo ">>> Re-using existing virtual environment in $VENV_DIR"
fi
# shellcheck disable=SC1091
if [ -f "$VENV_DIR/bin/activate" ]; then source "$VENV_DIR/bin/activate"; else source "$VENV_DIR/Scripts/activate"; fi
python -m pip install --upgrade pip setuptools wheel

# --- 3. Install PyTorch first (so the CPU/CUDA choice is respected) -------
TORCH_PKGS=$(grep -E '^torch==' "$REPO_ROOT/requirements.txt")
if [ -n "$TORCH_INDEX" ]; then
    echo ">>> Installing CPU-only PyTorch"
    # shellcheck disable=SC2086
    pip install $TORCH_PKGS --index-url "$TORCH_INDEX"
else
    echo ">>> Installing PyTorch (default build)"
    # shellcheck disable=SC2086
    pip install $TORCH_PKGS
fi

# --- 4. Install the remaining requirements and the Flower app -------------
echo ">>> Installing requirements.txt"
pip install -r "$REPO_ROOT/requirements.txt"
echo ">>> Installing the Flower app (fl_app/) in editable mode"
pip install -e "$REPO_ROOT/fl_app"

# --- 5. Sanity check -------------------------------------------------------
python - <<'EOF'
import flwr, numpy, pandas, scipy, torch
import aitdm_fl.task  # noqa: F401  (the Flower app is importable)
print("\n=== Environment ready ===")
print(f"flwr   {flwr.__version__}")
print(f"torch  {torch.__version__}  (CUDA available: {torch.cuda.is_available()})")
print(f"numpy  {numpy.__version__}, pandas {pandas.__version__}, scipy {scipy.__version__}")
EOF

echo
echo "Activate the environment in every new terminal with:"
echo "    source $VENV_DIR/bin/activate      (Git Bash on Windows: $VENV_DIR/Scripts/activate)"

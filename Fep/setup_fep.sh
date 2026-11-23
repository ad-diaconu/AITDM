#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-time setup on fep (run it on the login node, from the repository root).
#
#   bash Fep/setup_fep.sh                # .venv with PyTorch (CUDA): runs on CPU and GPU nodes
#   bash Fep/setup_fep.sh --cpu          # .venv with CPU-only PyTorch (CPU partitions only)
#   bash Fep/setup_fep.sh --container    # no .venv: use the apptainer image + venv that produced
#                                        # the results in outputs/ (see Fep/01_environment_setup.md)
#
# 1. Creates your project folder (default ~/AITDM_Project, change it with
#    AITDM_PROJECT=/other/path) with data/, logs/ and slurm/.
# 2. Links data/ and logs/ of this clone to it, so the data survives if you delete or
#    re-clone the repository. outputs/ is NOT linked: the results are committed to git.
#    An existing data/ folder of an older clone is moved there first.
# 3. Creates .venv with scripts/setup_env.sh (if it does not exist yet), or with
#    --container checks the image and the container venv.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AITDM_PROJECT="${AITDM_PROJECT:-$HOME/AITDM_Project}"
AITDM_IMAGE="${AITDM_IMAGE:-$HOME/NNProj/convnext.sif}"
AITDM_PYTHON="${AITDM_PYTHON:-$HOME/.venvs/aitdm-fl/bin/python}"

CONTAINER=0
case "${1:-}" in
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    --container) CONTAINER=1; shift ;;
esac

echo ">>> Project folder: $AITDM_PROJECT"
mkdir -p "$AITDM_PROJECT"/{data,logs,slurm}

for name in data logs; do
    link="$REPO_ROOT/$name"
    target="$AITDM_PROJECT/$name"
    if [ -L "$link" ]; then
        ln -sfn "$target" "$link"
    elif [ -d "$link" ]; then
        # A real folder from an earlier clone / run: move its content, then link
        echo ">>> Moving existing $name/ to $target"
        cp -a "$link/." "$target/"
        rm -rf "$link"
        ln -s "$target" "$link"
    else
        ln -s "$target" "$link"
    fi
    echo "    $name/ -> $target"
done

if [ "$CONTAINER" -eq 1 ]; then
    [ -f "$AITDM_IMAGE" ] || { echo "ERROR: no image $AITDM_IMAGE (set AITDM_IMAGE=...)" >&2; exit 1; }
    if [ ! -x "$AITDM_PYTHON" ]; then
        echo ">>> Creating the container venv ${AITDM_PYTHON%/bin/python} (on top of the image's PyTorch)"
        apptainer exec "$AITDM_IMAGE" python -m venv --system-site-packages "${AITDM_PYTHON%/bin/python}"
    fi
    echo ">>> Installing Flower into the container venv"
    apptainer exec "$AITDM_IMAGE" "$AITDM_PYTHON" -m pip install -q "flwr[simulation,dp]==1.41.0" scipy pandas scikit-learn matplotlib
    apptainer exec "$AITDM_IMAGE" "$AITDM_PYTHON" -c \
        "import flwr, torch; print('flwr', flwr.__version__, '| torch', torch.__version__)"
    NEXT="export AITDM_IMAGE=$AITDM_IMAGE   # add it to ~/.bashrc; every job then uses the container
    bash Fep/submit.sh smoke_test"
elif [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    echo ">>> .venv exists already (delete it to reinstall)"
    NEXT="bash Fep/submit.sh smoke_test"
else
    bash "$REPO_ROOT/scripts/setup_env.sh" "$@"
    NEXT="bash Fep/submit.sh smoke_test"
fi

cat <<EOF

=== fep setup done ===
Data lives in $AITDM_PROJECT/data, Slurm logs in $AITDM_PROJECT/slurm.
Next, check that everything works on a compute node (a few minutes):
    $NEXT
    squeue --me
EOF

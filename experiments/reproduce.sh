#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Re-creates every result in outputs/ from the raw WISDM archive, step by step.
# Finished runs are skipped, so the script can be stopped and restarted at any time.
#
#   bash experiments/reproduce.sh                  # everything (sequential: many GPU hours)
#   bash experiments/reproduce.sh data             # raw archive -> windows -> all partitions
#   bash experiments/reproduce.sh central          # centralised baseline, 2 models x 5 folds x 3 seeds
#   bash experiments/reproduce.sh fl [RUN LISTS]   # federated runs (default: every list in experiments/runs/)
#   bash experiments/reproduce.sh summary          # summaries and HTML reports
#
# On fep run the expensive steps as array jobs instead (parallel, Fep/02_running_on_slurm.md):
#   bash Fep/submit.sh centralised_array
#   bash Fep/submit.sh --array=0-8%2 fl_array experiments/runs/fl_primary.txt
# and this script only for `data` and `summary` (bash Fep/submit.sh reproduce data).
#
# WISDM_ARCHIVE: the WISDM_at_latest.tar.gz to install (default ~/AITDM_Project/WISDM_at_latest.tar.gz;
# downloaded from the WISDM Lab when it does not exist).
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=(${PY:-python})
STEP="${1:-all}"
[ $# -gt 0 ] && shift
want() { [ "$STEP" = all ] || [ "$STEP" = "$1" ]; }

if want data; then
    # the raw archive is only needed when the windows are missing (they can also be copied in: 22 MB)
    if [ ! -f data/wisdm/windows.npz ]; then
        ARCHIVE="${WISDM_ARCHIVE:-$HOME/AITDM_Project/WISDM_at_latest.tar.gz}"
        if [ ! -f data/raw/wisdm/WISDM_at_v2.0_raw.txt ]; then
            if [ -f "$ARCHIVE" ]; then
                "${PY[@]}" data_splitting/get_data.py --archive "$ARCHIVE"
            else
                "${PY[@]}" data_splitting/get_data.py --download
            fi
        fi
        "${PY[@]}" data_splitting/prepare_wisdm.py --input data/raw/wisdm/WISDM_at_v2.0_raw.txt --out data/wisdm
    fi
    # the splits of every experiment in outputs/wisdm_fl (seed 42, as in meta.json)
    for sk in "s0 1" "s0 3" "s0 5" "s1 3" "s1 5" "s2 3" "s2 5" "s3 3" "s3 5"; do
        set -- $sk
        [ -f "data/partitions/wisdm/$1_K$2/fold4/meta.json" ] ||
            "${PY[@]}" data_splitting/split_data.py --splits "$1" --clients "$2"
    done
fi

if want central; then
    for model in fcn lstm; do
        for fold in 0 1 2 3 4; do
            "${PY[@]}" -u experiments/centralised.py --data data/wisdm/windows.npz --model "$model" \
                --fold "$fold" --seeds 1 2 3 --out outputs/wisdm
        done
    done
fi

if want fl; then
    LISTS=("$@")
    [ ${#LISTS[@]} -gt 0 ] || LISTS=(experiments/runs/fl_sanity.txt experiments/runs/fl_primary.txt
                                     experiments/runs/fl_secondary.txt experiments/runs/fl_lstm_r80.txt)
    for list in "${LISTS[@]}"; do
        grep -v -E '^\s*(#|$)' "$list" | while read -r line; do
            eval "ARGV=($line)"  # parsed like a command line: quoted --config values work
            "${PY[@]}" -u experiments/run_seeds.py "${ARGV[@]}" < /dev/null
        done
    done
fi

if want summary; then
    "${PY[@]}" experiments/summarize.py --results outputs/wisdm --meta data/wisdm/meta.json
    if [ -d outputs/wisdm_fl ]; then "${PY[@]}" experiments/summarize_fl.py; fi
fi

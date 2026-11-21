#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# End-to-end smoke test on synthetic WISDM-format data (no download, nothing written to
# data/ or outputs/): prepare -> split -> centralised -> federated (FedAvg, FedBN, a robust
# strategy with an attacker, DP) -> summaries -> explainability -> one `flwr run` simulation.
#
#   bash tests/smoke_test.sh          (a few minutes on CPU; run by the CI and by
#                                      bash Fep/submit.sh smoke_test on a compute node)
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/_common.sh
PY=(${PY:-python})

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export PYTHONUTF8=1
q() { "$@" > "$WORK/last.log" 2>&1 || { tail -40 "$WORK/last.log"; echo "SMOKE TEST FAILED: $*" >&2; exit 1; }; }
need() { [ -f "$1" ] || { echo "SMOKE TEST FAILED: missing $1" >&2; exit 1; }; }

echo ">>> data: synthetic raw file -> windows -> partitions (s1 and s2, K=3)"
q "${PY[@]}" tests/synthetic_wisdm.py "$WORK/raw/WISDM_at_v2.0_raw.txt"
q "${PY[@]}" data_splitting/prepare_wisdm.py --input "$WORK/raw/WISDM_at_v2.0_raw.txt" --out "$WORK/wisdm"
q "${PY[@]}" data_splitting/split_data.py --windows "$WORK/wisdm/windows.npz" --out "$WORK/parts" \
    --splits s1 --clients 3
q "${PY[@]}" data_splitting/split_data.py --windows "$WORK/wisdm/windows.npz" --out "$WORK/parts" \
    --splits s1 --clients 3 --label-noise 0,0,0.5 --sensor-skew
need "$WORK/parts/s1_K3/fold0/client_2.npz"
need "$WORK/parts/s1_K3-noise0-0-0.5_skew/fold0/meta.json"

echo ">>> centralised baseline: fcn, 1 epoch, folds 0-1"
for f in 0 1; do
    q "${PY[@]}" experiments/centralised.py --data "$WORK/wisdm/windows.npz" --model fcn --fold "$f" \
        --seeds 1 --epochs 1 --out "$WORK/out/wisdm"
done
need "$WORK/out/wisdm/fcn/fold0/seed1/model.pt"

echo ">>> federated: fedavg, fedbn, krum + sign-flip attacker, dp, label-noise partitions (2 rounds)"
FL=(experiments/run_seeds.py --partitions-root "$WORK/parts" --out "$WORK/out/wisdm_fl" --split s1
    --clients 3 --model fcn --folds 0 1 --seeds 1 --rounds 2 --patience 1 --gpus 0 --cpus 2)
q "${PY[@]}" "${FL[@]}" --strategy fedavg
q "${PY[@]}" "${FL[@]}" --strategy fedbn
q "${PY[@]}" "${FL[@]}" --strategy krum --local-bn --config "malicious-clients='2' attack='sign-flip'" --tag signflip
q "${PY[@]}" "${FL[@]}" --strategy fedavg --config "dp-noise-multiplier=0.5" --tag dp
q "${PY[@]}" "${FL[@]}" --strategy fedavg --partition-tag noise0-0-0.5_skew
R="$WORK/out/wisdm_fl/s1_K3"
need "$R/fcn-fedbn/fold1/seed1/results.json"
need "$R/fcn-krum-localbn-signflip/fold0/seed1/results.json"
"${PY[@]}" -c "import json, sys; r = json.load(open(sys.argv[1])); assert r['privacy']['epsilon'] > 0, r.get('privacy')" \
    "$R/fcn-fedavg-dp/fold0/seed1/results.json"

echo ">>> summaries"
q "${PY[@]}" experiments/summarize.py --results "$WORK/out/wisdm" --meta "$WORK/wisdm/meta.json"
q "${PY[@]}" experiments/summarize_fl.py --fl "$WORK/out/wisdm_fl" --central "$WORK/out/wisdm" --partitions "$WORK/parts"
need "$WORK/out/wisdm_fl/report.html"

echo ">>> explainability"
q "${PY[@]}" explainability/error_analysis.py --run "$R/fcn-fedavg/fold0/seed1" --partitions "$WORK/parts/s1_K3/fold0"
q "${PY[@]}" explainability/feature_importance.py --run "$R/fcn-fedavg/fold0/seed1" \
    --partitions "$WORK/parts/s1_K3/fold0" --group-by segment --repeats 1
need "$R/fcn-fedavg/fold0/seed1/importance_val_segment.png"

if [ -z "${SMOKE_SKIP_FLWR_RUN:-}" ]; then
    echo ">>> flwr run (scripts/run_simulation.sh), 1 round"
    SAVE_DIR="$WORK/sim" q bash scripts/run_simulation.sh "$WORK/parts/s1_K3/fold0" "num-server-rounds=1"
    need "$WORK/sim/results.json"
fi

echo ">>> SMOKE TEST PASSED"

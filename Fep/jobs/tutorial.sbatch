#!/usr/bin/env bash
#SBATCH --job-name=aitdm-tutorial
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=00:20:00
# ---------------------------------------------------------------------------
# Tutorial experiment E0 (docs/tutorial.md): the whole pipeline on the real WISDM data in a
# few minutes on one GPU, for a demo. Fold 0, seed 1 only:
#   1. centralised FCN, 5 epochs                        -> outputs/tutorial/central/
#   2. FedAvg and FedBN, S2 (user-Dirichlet) K = 5, 10 rounds -> outputs/tutorial/fl/
#   3. a table of both next to the Stage 1 numbers        (experiments/tutorial.py)
#
#   bash Fep/submit.sh tutorial
#   bash Fep/submit.sh -p ucsx tutorial          # if dgxa100 is busy (squeue -p dgxa100)
#
# outputs/tutorial/ is git-ignored and emptied at the start of every submission, so the demo
# always trains for real. Needs data/wisdm/windows.npz (Fep/setup_fep.sh); the S2 K = 5
# partitions are made here if missing. Not a result: one fold, one seed, a short budget.
# ---------------------------------------------------------------------------
source Fep/jobs/_job_common.sh tutorial "$@"
OUT=outputs/tutorial
rm -rf "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
t0=$(date +%s)

if [ ! -f data/partitions/wisdm/s2_K5/fold0/meta.json ]; then
    echo ">>> 0/3 S2 K = 5 partitions (once, seed 42)"
    "${PY[@]}" -u data_splitting/split_data.py --splits s2 --clients 5
fi

echo ">>> 1/3 centralised FCN, fold 0, seed 1, 5 epochs"
"${PY[@]}" -u experiments/centralised.py --data data/wisdm/windows.npz --model fcn --fold 0 \
    --seeds 1 --epochs 5 --out "$OUT/central"

for strategy in fedavg fedbn; do
    echo ">>> 2/3 $strategy, S2 K = 5, fold 0, seed 1, 10 rounds"
    "${PY[@]}" -u experiments/run_seeds.py --split s2 --clients 5 --model fcn --strategy "$strategy" \
        --folds 0 --seeds 1 --rounds 10 --out "$OUT/fl"
done

echo ">>> 3/3 summary"
"${PY[@]}" experiments/tutorial.py --out "$OUT"
echo "=== done $(date), $(( $(date +%s) - t0 )) s"

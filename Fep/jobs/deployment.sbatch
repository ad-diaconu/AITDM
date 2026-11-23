#!/usr/bin/env bash
#SBATCH --job-name=aitdm-deployment
#SBATCH --partition=haswell
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
# ---------------------------------------------------------------------------
# Deployment mode inside ONE job on ONE partition: a SuperLink and one
# SuperNode per client, each a separate Slurm task (process), possibly on
# different nodes of the partition. They talk over the network, by IP address,
# exactly like separate machines (see Fep/02_running_on_slurm.md, section 3).
# With --gres=gpu:N (and -N 1) all components share the GPUs of the node.
#
#   bash Fep/submit.sh deployment data/partitions/wisdm/s2_K5/fold0              # 5 clients + 1
#   bash Fep/submit.sh deployment data/partitions/wisdm/s2_K5/fold0 "num-server-rounds=10"
#   bash Fep/submit.sh -n 6 deployment data/partitions/wisdm/s1_K5/fold0   # 1 + 5 tasks (= the default)
#   bash Fep/submit.sh -n 4 deployment data/partitions/wisdm/s1_K3/fold0          # 1 + 3 tasks
#
# --ntasks must be (number of clients + 1). The logs of every component go to
# logs/<jobid>/ (i.e. ~/AITDM_Project/logs), results to outputs/deploy-*.
# ---------------------------------------------------------------------------
source Fep/jobs/_job_common.sh deployment "$@"

PARTITIONS="$(abspath "${1:-$DEFAULT_PARTITIONS}")"
EXTRA_CONFIG="${2:-}"
check_partitions "$PARTITIONS"
NUM_CLIENTS="$(num_clients_of "$PARTITIONS")"
if [ "$SLURM_NTASKS" -lt $((NUM_CLIENTS + 1)) ]; then
    echo "ERROR: $NUM_CLIENTS clients need --ntasks=$((NUM_CLIENTS + 1)) (1 SuperLink + 1 per client)," >&2
    echo "       this job has $SLURM_NTASKS. Submit with: bash Fep/submit.sh -n $((NUM_CLIENTS + 1)) deployment ..." >&2
    exit 1
fi
LOG_DIR="$REPO_ROOT/logs/$SLURM_JOB_ID"
mkdir -p "$LOG_DIR"

# The SuperLink runs on this node (the first of the job). The SuperNodes connect
# to its IP address and to the port of this job (scripts/_common.sh: other jobs
# may share the node, so not the fixed 8000).
SUPERLINK_NODE="$(hostname -s)"
SUPERLINK_IP="$(getent ahostsv4 "$(hostname)" | awk 'NR == 1 {print $1}')"
export SUPERLINK_ADDRESS="$SUPERLINK_IP:$SUPERLINK_PORT"
export CONNECTION_NAME="slurm-$SLURM_JOB_ID"
echo "    nodes: $SLURM_JOB_NODELIST, SuperLink on $SUPERLINK_NODE ($SUPERLINK_ADDRESS)"

PIDS=()
cleanup() {
    echo ">>> Stopping SuperNodes and SuperLink"
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Every component is one task (a "step") of the job; steps do not count as jobs,
# so the whole federation uses ONE of your running jobs.
# CPU only: `srun --exact` gives every component its own CPUs.
# With GPUs (--gres=gpu:N, needs -N 1): all components share the node's GPUs
# (`--overlap`), as in a simulation on one machine.
NUM_GPUS="${SLURM_GPUS_ON_NODE:-0}"
if [ "$NUM_GPUS" -gt 0 ] && [ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ]; then
    echo "ERROR: with --gres=gpu the deployment job must run on one node: add -N 1" >&2
    exit 1
fi
step() {  # step LOGFILE COMMAND...
    local log="$1"
    shift
    if [ "$NUM_GPUS" -gt 0 ]; then
        srun --ntasks=1 --nodes=1 --overlap --gres="gpu:$NUM_GPUS" "$@" > "$log" 2>&1 &
    else
        srun --ntasks=1 --nodes=1 --exact --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
            "$@" > "$log" 2>&1 &
    fi
    PIDS+=($!)
}

echo ">>> SuperLink                 -> logs/$SLURM_JOB_ID/superlink.log"
step "$LOG_DIR/superlink.log" --nodelist="$SUPERLINK_NODE" \
    env HOST=0.0.0.0 PORT="$SUPERLINK_PORT" bash scripts/start_superlink.sh
python - "$SUPERLINK_IP" "$SUPERLINK_PORT" <<'EOF'
import socket, sys, time
for _ in range(120):
    try:
        socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1).close()
        break
    except OSError:
        time.sleep(1)
else:
    sys.exit("ERROR: the SuperLink did not start, see its log")
EOF

for ((i = 0; i < NUM_CLIENTS; i++)); do
    echo ">>> SuperNode $i               -> logs/$SLURM_JOB_ID/supernode_$i.log"
    step "$LOG_DIR/supernode_$i.log" \
        bash scripts/start_supernode.sh "$i" "$PARTITIONS" "$SUPERLINK_ADDRESS"
done
sleep 5
echo ">>> Where every component runs (step 0 = SuperLink, 1.. = SuperNodes):"
squeue -h -s -j "$SLURM_JOB_ID" -o '    step %i on %N' | grep -v -e batch -e extern || true

echo ">>> flwr run (connection '$CONNECTION_NAME' -> $SUPERLINK_ADDRESS)"
bash scripts/submit_run.sh "$PARTITIONS" "$EXTRA_CONFIG"

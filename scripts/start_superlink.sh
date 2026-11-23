#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# TERMINAL 1 – the server side (SuperLink). Keep this terminal open.
#
# Usage:
#   bash scripts/start_superlink.sh
#
# The SuperLink listens on 127.0.0.1:8000. SuperNodes (clients) connect to it
# and `flwr run` submits the app to it. To accept clients from OTHER machines,
# start it with:  HOST=0.0.0.0 bash scripts/start_superlink.sh
# ---------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-$SUPERLINK_PORT}"   # 8000, or the port of your Slurm job
echo ">>> Starting SuperLink on $HOST:$PORT (Ctrl+C to stop)"
exec flower-superlink \
    --insecure \
    --host "$HOST" \
    --port "$PORT" \
    --disable-runtime-dependency-installation

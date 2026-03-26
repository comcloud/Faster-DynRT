#!/usr/bin/env bash
set -euo pipefail

REMOTE="root@connect.westb.seetacloud.com"
PORT="12426"
REMOTE_BASE="/root/autodl-tmp/ray/projects/dynrt_bridge_main/auto_runs/background_controller"
LOCAL_BASE="$(cd "$(dirname "$0")" && pwd)"

scp -P "${PORT}" "${REMOTE}:${REMOTE_BASE}/data_report.json" "${LOCAL_BASE}/data_report.remote.json"
scp -P "${PORT}" "${REMOTE}:${REMOTE_BASE}/controller_state.json" "${LOCAL_BASE}/controller_state.remote.json"
scp -P "${PORT}" "${REMOTE}:${REMOTE_BASE}/events.jsonl" "${LOCAL_BASE}/events.remote.jsonl"

python "${LOCAL_BASE}/build_remote_data_dashboard.py"
echo "Dashboard updated: ${LOCAL_BASE}/remote_data_dashboard.html"

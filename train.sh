#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/train_logs}"
PYTHON="${PYTHON:-python}"
command -v "$PYTHON" > /dev/null
mkdir -p -- "$LOG_DIR"

TIME_STR=$(date +'%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/run_${TIME_STR}_$$.log"

nohup "$PYTHON" -u "${SCRIPT_DIR}/train.py" --config "${SCRIPT_DIR}/cadrec_config.yaml" "$@" > "$LOG_FILE" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "${LOG_FILE%.log}.pid"
printf 'Started training (PID %s).\nLog: %s\n' "$PID" "$LOG_FILE"

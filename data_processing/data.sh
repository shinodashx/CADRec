#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/datalogs}"
PYTHON="${PYTHON:-python}"
command -v "$PYTHON" > /dev/null
mkdir -p -- "$LOG_DIR"

TIME_STR=$(date +'%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/data_${TIME_STR}_$$.log"

nohup "$PYTHON" -u "${SCRIPT_DIR}/batch_json2cadquery.py" "$@" > "$LOG_FILE" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "${LOG_FILE%.log}.pid"
printf 'Started preprocessing (PID %s).\nLog: %s\n' "$PID" "$LOG_FILE"

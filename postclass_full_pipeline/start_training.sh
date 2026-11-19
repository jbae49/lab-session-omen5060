#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RESULT_NAME" >&2
  exit 2
fi

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="$1"
STATE_ROOT="$LAB_ROOT/outputs/postclass_training"
LATEST_FILE="$STATE_ROOT/latest_run_dir.txt"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_STATE="$STATE_ROOT/${NAME}_$STAMP"
mkdir -p "$RUN_STATE"

if [[ -f "$LATEST_FILE" ]]; then
  PREVIOUS_RUN="$(<"$LATEST_FILE")"
  if [[ -f "$PREVIOUS_RUN/train.pid" ]]; then
    PREVIOUS_PID="$(<"$PREVIOUS_RUN/train.pid")"
    if kill -0 "$PREVIOUS_PID" 2>/dev/null; then
      echo "이미 수업 후 학습이 실행 중입니다 (PID $PREVIOUS_PID)." >&2
      exit 1
    fi
  fi
fi

printf '%s\n' "$RUN_STATE" > "$LATEST_FILE"
printf '%s\n' "$NAME" > "$RUN_STATE/result_name.txt"
LOG_FILE="$RUN_STATE/train.log"
COMMAND=(env GPU="${GPU:-0}" LOGGER="${LOGGER:-tensorboard}" \
  bash "$LAB_ROOT/postclass_full_pipeline/train.sh" full "$NAME")
if command -v systemd-inhibit >/dev/null 2>&1 && systemd-inhibit --list >/dev/null 2>&1; then
  COMMAND=(systemd-inhibit --what=sleep:idle --mode=block --why="postclass G1 training" "${COMMAND[@]}")
fi
nohup "${COMMAND[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$RUN_STATE/train.pid"
sleep 2
if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
  echo "학습이 시작 직후 종료되었습니다: $LOG_FILE" >&2
  tail -n 40 "$LOG_FILE" >&2 || true
  exit 1
fi
echo "학습 시작: PID=$TRAIN_PID"
echo "로그: $LOG_FILE"
echo "확인: bash postclass_full_pipeline/training_status.sh"


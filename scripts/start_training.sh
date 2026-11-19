#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-full}"
MOTION="${2:-$LAB_ROOT/outputs/jump_retarget_lab/jump_retarget_lab_g1_29dof_50hz.npz}"
GPU="${GPU:-0}"
LOGGER="${LOGGER:-tensorboard}"
STATE_ROOT="$LAB_ROOT/outputs/training"
LATEST_FILE="$STATE_ROOT/latest_run_dir.txt"

if [[ ! -f "$MOTION" ]]; then
  echo "Reference motion이 없습니다: $MOTION" >&2
  echo "먼저 실행하세요: bash scripts/run_retarget_lab.sh" >&2
  exit 1
fi

if [[ -f "$LATEST_FILE" ]]; then
  PREVIOUS_RUN="$(<"$LATEST_FILE")"
  if [[ -f "$PREVIOUS_RUN/train.pid" ]]; then
    PREVIOUS_PID="$(<"$PREVIOUS_RUN/train.pid")"
    if kill -0 "$PREVIOUS_PID" 2>/dev/null; then
      echo "이미 학습이 실행 중입니다 (PID $PREVIOUS_PID)." >&2
      echo "상태 확인: bash scripts/training_status.sh" >&2
      exit 1
    fi
  fi
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_STATE="$STATE_ROOT/$STAMP"
mkdir -p "$RUN_STATE"
LOG_FILE="$RUN_STATE/train.log"
printf '%s\n' "$RUN_STATE" > "$LATEST_FILE"
printf '%s\n' "$MOTION" > "$RUN_STATE/motion_file.txt"
printf '%s\n' "$PROFILE" > "$RUN_STATE/profile.txt"

TRAIN_COMMAND=(env GPU="$GPU" LOGGER="$LOGGER" bash "$LAB_ROOT/scripts/train_mimic.sh" "$PROFILE" "$MOTION")
if command -v systemd-inhibit >/dev/null 2>&1 && systemd-inhibit --list >/dev/null 2>&1; then
  TRAIN_COMMAND=(systemd-inhibit --what=sleep:idle --mode=block --why="G1 mimic training" "${TRAIN_COMMAND[@]}")
fi

nohup "${TRAIN_COMMAND[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$RUN_STATE/train.pid"

sleep 2
if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
  echo "학습이 시작 직후 종료되었습니다. 로그를 확인하세요: $LOG_FILE" >&2
  tail -n 40 "$LOG_FILE" >&2 || true
  exit 1
fi

echo "장기 학습을 백그라운드에서 시작했습니다."
echo "  PID:       $TRAIN_PID"
echo "  profile:   $PROFILE"
echo "  GPU:       $GPU"
echo "  logger:    $LOGGER"
echo "  reference: $MOTION"
echo "  log:       $LOG_FILE"
echo
echo "터미널은 닫아도 되지만 노트북 전원과 어댑터를 유지하세요."
echo "다음 날 확인: bash scripts/training_status.sh"

#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LATEST_FILE="$LAB_ROOT/outputs/training/latest_run_dir.txt"
LOG_ROOT="$LAB_ROOT/repos/unitree_rl_mjlab/logs/rsl_rl/g1_tracking"

if [[ ! -f "$LATEST_FILE" ]]; then
  echo "시작 기록이 없습니다. 먼저 실행하세요: bash scripts/start_training.sh full" >&2
  exit 1
fi

RUN_STATE="$(<"$LATEST_FILE")"
PID_FILE="$RUN_STATE/train.pid"
LOG_FILE="$RUN_STATE/train.log"
if [[ ! -f "$PID_FILE" ]]; then
  echo "PID 파일이 없습니다: $PID_FILE" >&2
  exit 1
fi

TRAIN_PID="$(<"$PID_FILE")"
if kill -0 "$TRAIN_PID" 2>/dev/null; then
  echo "상태: 실행 중 (PID $TRAIN_PID)"
else
  echo "상태: 프로세스 종료됨 (완료 또는 오류)"
fi

echo
echo "=== 최근 학습 로그 ==="
tail -n 30 "$LOG_FILE" 2>/dev/null || echo "아직 로그가 없습니다."

echo
echo "=== 최신 checkpoint ==="
CHECKPOINT_LINE="$(find "$LOG_ROOT" -type f -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | sed -n '1p')"
if [[ -n "$CHECKPOINT_LINE" ]]; then
  echo "${CHECKPOINT_LINE#* }"
else
  echo "아직 저장된 checkpoint가 없습니다."
fi

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=name,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader 2>/dev/null || echo "nvidia-smi 정보를 읽을 수 없습니다."

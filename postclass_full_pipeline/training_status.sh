#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LATEST_FILE="$LAB_ROOT/outputs/postclass_training/latest_run_dir.txt"
LOG_ROOT="$LAB_ROOT/repos/unitree_rl_mjlab/logs/rsl_rl/postclass_g1_tracking"
if [[ ! -f "$LATEST_FILE" ]]; then
  echo "수업 후 학습 시작 기록이 없습니다." >&2
  exit 1
fi
RUN_STATE="$(<"$LATEST_FILE")"
TRAIN_PID="$(<"$RUN_STATE/train.pid")"
if kill -0 "$TRAIN_PID" 2>/dev/null; then
  echo "상태: 실행 중 (PID $TRAIN_PID)"
else
  echo "상태: 종료됨 (완료 또는 오류)"
fi
echo "=== 최근 로그 ==="
tail -n 30 "$RUN_STATE/train.log" 2>/dev/null || true
echo "=== 최신 checkpoint ==="
find "$LOG_ROOT" -type f -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr | sed -n '1s/^[^ ]* //p'
echo "=== GPU ==="
nvidia-smi --query-gpu=name,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader 2>/dev/null || true


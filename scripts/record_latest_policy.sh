#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOTION="${1:-$LAB_ROOT/outputs/jump_retarget_lab/jump_retarget_lab_g1_29dof_50hz.npz}"
GPU="${GPU:-0}"
LOG_ROOT="$LAB_ROOT/repos/unitree_rl_mjlab/logs/rsl_rl/g1_tracking"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$LAB_ROOT/outputs/policy_rollouts/$STAMP"
ROLLOUT="$OUTPUT_DIR/policy_rollout.npz"
VIDEO="$OUTPUT_DIR/policy_rollout.mp4"

if [[ ! -f "$MOTION" ]]; then
  echo "Reference motion이 없습니다: $MOTION" >&2
  exit 1
fi

CHECKPOINT_LINE="$(find "$LOG_ROOT" -type f -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | sed -n '1p')"
if [[ -z "$CHECKPOINT_LINE" ]]; then
  echo "Checkpoint를 찾지 못했습니다: $LOG_ROOT" >&2
  exit 1
fi
CHECKPOINT="${CHECKPOINT_LINE#* }"
mkdir -p "$OUTPUT_DIR"

cd "$LAB_ROOT/repos/unitree_rl_mjlab"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl \
  conda run --no-capture-output -n unitree_mjlab_omen5060 \
  python scripts/record_policy_motion.py \
  --task ClassLab-G1-Tracking-No-State-Estimation \
  --motion-file "$MOTION" --checkpoint-file "$CHECKPOINT" \
  --output "$ROLLOUT" --device cuda:0

MUJOCO_GL=egl conda run --no-capture-output -n unitree_mjlab_omen5060 \
  python scripts/render_motion_npz.py \
  --motion-file "$ROLLOUT" --output "$VIDEO" --width 960 --height 540

echo
echo "최신 checkpoint: $CHECKPOINT"
echo "정책 rollout:    $ROLLOUT"
echo "직접 확인할 영상: $VIDEO"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RESULT_NAME" >&2
  exit 2
fi

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="$1"
GPU="${GPU:-0}"
MOTION="$LAB_ROOT/outputs/postclass/$NAME/mjlab/${NAME}_g1_29dof_50hz.npz"
LOG_ROOT="$LAB_ROOT/repos/unitree_rl_mjlab/logs/rsl_rl/postclass_g1_tracking"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$LAB_ROOT/outputs/postclass_policy_rollouts/${NAME}_$STAMP"
ROLLOUT="$OUTPUT_DIR/policy_rollout.npz"
VIDEO="$OUTPUT_DIR/policy_rollout.mp4"

if [[ ! -f "$MOTION" ]]; then
  echo "Reference가 없습니다: $MOTION" >&2
  exit 1
fi
CHECKPOINT_LINE="$(find "$LOG_ROOT" -type f -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | sed -n '1p')"
if [[ -z "$CHECKPOINT_LINE" ]]; then
  echo "Checkpoint가 없습니다: $LOG_ROOT" >&2
  exit 1
fi
CHECKPOINT="${CHECKPOINT_LINE#* }"
mkdir -p "$OUTPUT_DIR"

cd "$LAB_ROOT/repos/unitree_rl_mjlab"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl \
  conda run --no-capture-output -n mjlab_postclass_omen5060 \
  python scripts/record_policy_motion.py \
  --task ClassLab-G1-Tracking-No-State-Estimation \
  --motion-file "$MOTION" --checkpoint-file "$CHECKPOINT" \
  --output "$ROLLOUT" --device cuda:0
MUJOCO_GL=egl conda run --no-capture-output -n mjlab_postclass_omen5060 \
  python scripts/render_motion_npz.py \
  --motion-file "$ROLLOUT" --output "$VIDEO" --width 960 --height 540

echo "최신 checkpoint: $CHECKPOINT"
echo "직접 확인할 영상: $VIDEO"

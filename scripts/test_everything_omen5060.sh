#!/usr/bin/env bash
set -euo pipefail

# One-command acceptance test after the user has downloaded SMPL/SMPL-X.
LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:-0}"
NAME="${NAME:-omen5060_acceptance}"
VIDEO="${VIDEO:-$LAB_ROOT/sample_data/input/jump.mov}"
MOTION="$LAB_ROOT/outputs/$NAME/mjlab/${NAME}_g1_29dof_50hz.npz"
ROLLOUT="$LAB_ROOT/outputs/$NAME/rl/${NAME}_smoke_rollout.npz"
ROLLOUT_MP4="$LAB_ROOT/outputs/$NAME/rl/${NAME}_smoke_rollout.mp4"
LOG_ROOT="$LAB_ROOT/repos/unitree_rl_mjlab/logs/rsl_rl/g1_tracking"
MARKER="$(mktemp)"
trap 'rm -f "$MARKER"' EXIT

echo "=== 1/4 파일·GPU 확인 ==="
python "$LAB_ROOT/scripts/doctor.py" --stage files
conda run -n phmr_omen5060 python "$LAB_ROOT/scripts/doctor.py" --stage prompthmr
conda run -n gmr_omen5060 python "$LAB_ROOT/scripts/doctor.py" --stage gmr
conda run -n unitree_mjlab_omen5060 python "$LAB_ROOT/scripts/doctor.py" --stage mjlab

echo "=== 2/4 영상 -> PromptHMR -> GMR -> MjLab -> 두 개의 MP4 ==="
GPU="$GPU" bash "$LAB_ROOT/scripts/run_jump_pipeline.sh" "$VIDEO" "$NAME"

echo "=== 3/4 32 env / 20 iteration RL smoke training ==="
GPU="$GPU" LOGGER=tensorboard bash "$LAB_ROOT/scripts/train_mimic.sh" smoke "$MOTION"
CHECKPOINT="$(find "$LOG_ROOT" -type f -name 'model_*.pt' -newer "$MARKER" -printf '%T@ %p\n' \
  | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -z "$CHECKPOINT" ]]; then
  echo "새 smoke checkpoint를 찾지 못했습니다: $LOG_ROOT" >&2
  exit 1
fi

echo "=== 4/4 새 checkpoint rollout + MP4 ==="
mkdir -p "$(dirname "$ROLLOUT")"
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
  --motion-file "$ROLLOUT" --output "$ROLLOUT_MP4" --width 960 --height 540

echo
echo "================== 테스트 성공 =================="
echo "아래 세 영상을 VLC 또는 파일 탐색기에서 꼭 직접 확인하세요!"
echo "1) 사람 mesh:  $LAB_ROOT/outputs/$NAME/prompthmr/overlay.mp4"
echo "2) G1 reference: $LAB_ROOT/outputs/$NAME/gmr/${NAME}_g1_reference.mp4"
echo "3) RL rollout:  $ROLLOUT_MP4"
echo "smoke checkpoint: $CHECKPOINT"
echo "주의: 20 iteration은 설치 검증용이므로 동작 품질은 낮은 것이 정상입니다."

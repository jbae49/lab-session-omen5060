#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
VIDEO="${1:-$LAB_ROOT/sample_data/input/jump.mov}"
NAME="${2:-jump_student}"
START_FRAME="${START_FRAME:-0}"
PERSON_RANK="${PERSON_RANK:-0}"
TARGET_X="${TARGET_X-426.5}"
TARGET_Y="${TARGET_Y-294.0}"

PHMR_OUT="$LAB_ROOT/outputs/$NAME/prompthmr"
SMPLX_OUT="$LAB_ROOT/outputs/$NAME/gmr/${NAME}_smplx.npz"
GMR_QPOS="$LAB_ROOT/outputs/$NAME/gmr/${NAME}_g1_29dof_qpos.npz"
MJLAB_OUT="$LAB_ROOT/outputs/$NAME/mjlab/${NAME}_g1_29dof_50hz.npz"

mkdir -p "$PHMR_OUT" "$(dirname "$SMPLX_OUT")" "$(dirname "$MJLAB_OUT")"

TARGET_ARGS=()
if [[ -n "$TARGET_X" && -n "$TARGET_Y" ]]; then
  TARGET_ARGS=(--target-x "$TARGET_X" --target-y "$TARGET_Y")
fi

conda run --no-capture-output -n phmr_omen5060 \
  python "$LAB_ROOT/scripts/01_extract_prompthmr.py" \
  --video "$VIDEO" --output "$PHMR_OUT" --static-camera \
  --start-frame "$START_FRAME" --person-rank "$PERSON_RANK" --num-people 1 \
  "${TARGET_ARGS[@]}" \
  --debug-masks

conda run --no-capture-output -n phmr_omen5060 \
  python "$LAB_ROOT/scripts/02_export_smplx_for_gmr.py" \
  --input "$PHMR_OUT/results.pkl" --output "$SMPLX_OUT" --video "$VIDEO"

conda run --no-capture-output -n phmr_omen5060 \
  python "$LAB_ROOT/scripts/05_render_prompthmr_overlay.py" \
  --video "$VIDEO" --results "$PHMR_OUT/results.pkl" \
  --output "$PHMR_OUT/overlay.mp4"

conda run --no-capture-output -n gmr_omen5060 \
  python "$LAB_ROOT/scripts/03_retarget_gmr.py" \
  --input "$SMPLX_OUT" --output "$GMR_QPOS" --fps 30

conda run --no-capture-output -n unitree_mjlab_omen5060 \
  python "$LAB_ROOT/scripts/04_prepare_mjlab_motion.py" \
  --input "$GMR_QPOS" --output "$MJLAB_OUT" --fps 50

(
  cd "$LAB_ROOT/repos/unitree_rl_mjlab"
  export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
  MUJOCO_GL=egl conda run --no-capture-output -n unitree_mjlab_omen5060 \
    python scripts/render_motion_npz.py \
    --motion-file "$MJLAB_OUT" \
    --output "$LAB_ROOT/outputs/$NAME/gmr/${NAME}_g1_reference.mp4" \
    --width 960 --height 540
)

echo "Pipeline complete: $MJLAB_OUT"
echo "꼭 직접 확인하세요:"
echo "  사람 mesh overlay: $PHMR_OUT/overlay.mp4"
echo "  G1 retarget 영상:   $LAB_ROOT/outputs/$NAME/gmr/${NAME}_g1_reference.mp4"

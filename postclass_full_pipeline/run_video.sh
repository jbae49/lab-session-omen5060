#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 VIDEO_PATH RESULT_NAME START_FRAME TARGET_X TARGET_Y" >&2
  echo "Example: $0 /absolute/path/dance.mp4 my_dance 30 720 350" >&2
  exit 2
fi

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO="$(realpath "$1")"
NAME="$2"
START_FRAME="$3"
TARGET_X="$4"
TARGET_Y="$5"
GPU="${GPU:-0}"
OUTPUT_ROOT="$LAB_ROOT/outputs/postclass/$NAME"
PHMR_OUT="$OUTPUT_ROOT/prompthmr"
SMPLX_OUT="$OUTPUT_ROOT/gmr/${NAME}_smplx.npz"
QPOS_OUT="$OUTPUT_ROOT/gmr/${NAME}_g1_29dof_qpos.npz"
REFERENCE_OUT="$OUTPUT_ROOT/mjlab/${NAME}_g1_29dof_50hz.npz"
REFERENCE_VIDEO="$OUTPUT_ROOT/gmr/${NAME}_g1_reference.mp4"

if [[ ! -f "$VIDEO" ]]; then
  echo "영상을 찾을 수 없습니다: $VIDEO" >&2
  exit 1
fi
if [[ ! "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo "RESULT_NAME은 영문/숫자로 시작하고 영문, 숫자, _, -만 사용할 수 있습니다." >&2
  exit 1
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "기존 결과를 보호하기 위해 중단합니다: $OUTPUT_ROOT" >&2
  echo "다른 RESULT_NAME을 사용하세요." >&2
  exit 1
fi

mkdir -p "$PHMR_OUT" "$(dirname "$SMPLX_OUT")" "$(dirname "$REFERENCE_OUT")"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "=== 1/6 RGB video -> selected SMPL-X track ==="
conda run --no-capture-output -n phmr_postclass_omen5060 \
  python "$LAB_ROOT/scripts/01_extract_prompthmr.py" \
  --video "$VIDEO" --output "$PHMR_OUT" --static-camera \
  --start-frame "$START_FRAME" --target-x "$TARGET_X" --target-y "$TARGET_Y" \
  --person-rank 0 --num-people 1 --debug-masks

echo "=== 2/6 raw PromptHMR -> GMR SMPL-X NPZ ==="
conda run --no-capture-output -n phmr_postclass_omen5060 \
  python "$LAB_ROOT/scripts/02_export_smplx_for_gmr.py" \
  --input "$PHMR_OUT/results.pkl" --output "$SMPLX_OUT" --video "$VIDEO"

echo "=== 3/6 original video + SMPL-X overlay ==="
conda run --no-capture-output -n phmr_postclass_omen5060 \
  python "$LAB_ROOT/scripts/05_render_prompthmr_overlay.py" \
  --video "$VIDEO" --results "$PHMR_OUT/results.pkl" --output "$PHMR_OUT/overlay.mp4"

echo "=== 4/6 SMPL-X -> G1 29DoF (GMR) ==="
conda run --no-capture-output -n gmr_postclass_omen5060 \
  python "$LAB_ROOT/scripts/03_retarget_gmr.py" \
  --input "$SMPLX_OUT" --output "$QPOS_OUT" --fps 30 --device cuda:0

echo "=== 5/6 G1 qpos -> 50Hz MjLab reference ==="
conda run --no-capture-output -n mjlab_postclass_omen5060 \
  python "$LAB_ROOT/scripts/04_prepare_mjlab_motion.py" \
  --input "$QPOS_OUT" --output "$REFERENCE_OUT" --fps 50

echo "=== 6/6 G1 reference MP4 ==="
(
  cd "$LAB_ROOT/repos/unitree_rl_mjlab"
  export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
  MUJOCO_GL=egl conda run --no-capture-output -n mjlab_postclass_omen5060 \
    python scripts/render_motion_npz.py \
    --motion-file "$REFERENCE_OUT" --output "$REFERENCE_VIDEO" --width 960 --height 540
)

echo
echo "전체 변환 완료. 학습 전에 두 영상을 직접 확인하세요:"
echo "  사람 overlay: $PHMR_OUT/overlay.mp4"
echo "  G1 reference: $REFERENCE_VIDEO"
echo "학습 reference: $REFERENCE_OUT"


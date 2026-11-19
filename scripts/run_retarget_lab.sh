#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="${1:-$LAB_ROOT/sample_data/input/jump_smplx_joint_targets_30fps.npz}"
NAME="${2:-jump_retarget_lab}"
GPU="${GPU:-0}"
QPOS="$LAB_ROOT/outputs/$NAME/${NAME}_g1_29dof_qpos.npz"
REFERENCE="$LAB_ROOT/outputs/$NAME/${NAME}_g1_29dof_50hz.npz"
VIDEO="$LAB_ROOT/outputs/$NAME/${NAME}_g1_retarget.mp4"

export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$LAB_ROOT/outputs/$NAME"

echo "=== 1/4 제공된 사람 joint motion 확인 ==="
conda run -n gmr_omen5060 python "$LAB_ROOT/scripts/doctor.py" --stage retarget

echo "=== 2/4 SMPL-X joint targets -> G1 29DoF (GMR) ==="
conda run --no-capture-output -n gmr_omen5060 \
  python "$LAB_ROOT/scripts/03_retarget_joint_targets_gmr.py" \
  --input "$INPUT" --output "$QPOS" --device cuda:0

echo "=== 3/4 30 Hz qpos -> 50 Hz MJLab reference ==="
conda run --no-capture-output -n unitree_mjlab_omen5060 \
  python "$LAB_ROOT/scripts/04_prepare_mjlab_motion.py" \
  --input "$QPOS" --output "$REFERENCE" --fps 50

echo "=== 4/4 지면이 포함된 G1 MP4 녹화 ==="
(
  cd "$LAB_ROOT/repos/unitree_rl_mjlab"
  export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
  MUJOCO_GL=egl conda run --no-capture-output -n unitree_mjlab_omen5060 \
    python scripts/render_motion_npz.py \
    --motion-file "$REFERENCE" --output "$VIDEO" --width 960 --height 540
)

echo
echo "================ 리타깃 실습 완료 ================"
echo "GMR qpos:       $QPOS"
echo "MJLab reference: $REFERENCE"
echo "직접 확인할 영상: $VIDEO"
echo "VLC나 파일 탐색기로 MP4를 꼭 재생해 보세요!"

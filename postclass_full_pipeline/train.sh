#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 2 ]]; then
  echo "Usage: $0 {smoke|quick|full} RESULT_NAME" >&2
  exit 2
fi

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="$1"
NAME="$2"
MOTION="$LAB_ROOT/outputs/postclass/$NAME/mjlab/${NAME}_g1_29dof_50hz.npz"
GPU="${GPU:-0}"
LOGGER="${LOGGER:-tensorboard}"

if [[ ! -f "$MOTION" ]]; then
  echo "학습 reference가 없습니다: $MOTION" >&2
  echo "run_video.sh를 먼저 실행하세요." >&2
  exit 1
fi

case "$PROFILE" in
  smoke) ENVS=32; ITERS=20 ;;
  quick) ENVS=128; ITERS=1000 ;;
  full) ENVS=256; ITERS=60000 ;;
  *) echo "PROFILE은 smoke, quick, full 중 하나여야 합니다." >&2; exit 2 ;;
esac

cd "$LAB_ROOT/repos/unitree_rl_mjlab"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n mjlab_postclass_omen5060 \
  python scripts/train.py ClassLab-G1-Tracking-No-State-Estimation \
  --motion-file "$MOTION" \
  --env.scene.num-envs "$ENVS" \
  --agent.max-iterations "$ITERS" \
  --agent.experiment-name postclass_g1_tracking \
  --agent.logger "$LOGGER" \
  --agent.run-name "postclass_${NAME}_${PROFILE}" \
  --gpu-ids '[0]'


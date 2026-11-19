#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-quick}"
MOTION="${2:-$LAB_ROOT/outputs/jump_retarget_lab/jump_retarget_lab_g1_29dof_50hz.npz}"
GPU="${GPU:-0}"
LOGGER="${LOGGER:-tensorboard}"

if [[ ! -f "$MOTION" ]]; then
  echo "Reference motion이 없습니다: $MOTION" >&2
  echo "먼저 실행하세요: bash scripts/run_retarget_lab.sh" >&2
  exit 1
fi

case "$PROFILE" in
  smoke) ENVS=32; ITERS=20 ;;
  quick) ENVS=128; ITERS=1000 ;;
  full) ENVS=256; ITERS=60000 ;;
  *) echo "Usage: $0 {smoke|quick|full} [motion.npz]" >&2; exit 2 ;;
esac

cd "$LAB_ROOT/repos/unitree_rl_mjlab"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n unitree_mjlab_omen5060 \
  python scripts/train.py ClassLab-G1-Tracking-No-State-Estimation \
  --motion-file "$MOTION" \
  --env.scene.num-envs "$ENVS" \
  --agent.max-iterations "$ITERS" \
  --agent.logger "$LOGGER" \
  --agent.run-name "lab_${PROFILE}_g1_29dof_$(basename "$MOTION" .npz)" \
  --gpu-ids '[0]'

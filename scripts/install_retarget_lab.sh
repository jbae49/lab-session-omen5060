#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--execute" ]]; then
  echo "수업용 GMR + MJLab 환경만 설치합니다."
  echo "PromptHMR, Google Drive weight, SMPL/SMPL-X model은 필요하지 않습니다."
  echo "실행: bash scripts/install_retarget_lab.sh --execute"
  exit 2
fi

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for COMMAND in conda git ffmpeg nvidia-smi; do
  if ! command -v "$COMMAND" >/dev/null 2>&1; then
    echo "필수 명령을 찾을 수 없습니다: $COMMAND" >&2
    exit 1
  fi
done
eval "$(conda shell.bash hook)"

ensure_env() {
  local env_name="$1"
  if conda run -n "$env_name" python -c "import sys" >/dev/null 2>&1; then
    echo "[resume] 기존 환경 재사용: $env_name"
  else
    conda create -n "$env_name" python=3.11 -y
  fi
  conda run -n "$env_name" python -m pip install --upgrade "pip<26" setuptools wheel
  conda run -n "$env_name" python -m pip install \
    torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
}

echo "[1/2] GMR retargeting 환경"
ensure_env gmr_omen5060
conda run -n gmr_omen5060 python -m pip install -e "$LAB_ROOT/repos/GMR"

echo "[2/2] MJLab reference/visualization 환경"
ensure_env unitree_mjlab_omen5060
conda run -n unitree_mjlab_omen5060 python -m pip install \
  -e "$LAB_ROOT/repos/unitree_rl_mjlab" \
  scipy "warp-lang==1.12.0" "mujoco==3.5.0" "mujoco-warp==3.5.0"
conda run -n unitree_mjlab_omen5060 python -m pip install tensorboard wandb

echo
echo "설치 완료. 다음 명령으로 수업 실습을 실행하세요:"
echo "  bash scripts/run_retarget_lab.sh"

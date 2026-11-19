#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--execute" ]]; then
  echo "수업 후 전체 파이프라인용 별도 Conda 환경 3개를 설치합니다."
  echo "기존 수업용 환경은 수정하지 않습니다."
  echo "실행: bash postclass_full_pipeline/install.sh --execute"
  exit 2
fi

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHMR="$LAB_ROOT/repos/PromptHMR"
for COMMAND in conda git wget unzip ffmpeg nvidia-smi; do
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
  conda run -n "$env_name" python -m pip install --upgrade "pip<26" setuptools wheel ninja
}

echo "[1/5] 독립 PromptHMR 환경"
ensure_env phmr_postclass_omen5060
conda install -n phmr_postclass_omen5060 -c nvidia/label/cuda-12.8.1 cuda-toolkit -y
conda run -n phmr_postclass_omen5060 python -m pip install \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
conda run -n phmr_postclass_omen5060 python -m pip install \
  torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
conda run -n phmr_postclass_omen5060 python -m pip install "numpy<2" -r "$PHMR/requirements.txt"
conda run -n phmr_postclass_omen5060 python -m pip install --no-build-isolation \
  "git+https://github.com/Arthur151/chumpy.git@ec5a20f3adc1a105c7a907cfed9a33d7dccbe95f"
conda run -n phmr_postclass_omen5060 python -m pip install xformers==0.0.31.post1 \
  --index-url https://download.pytorch.org/whl/cu128 --no-deps

echo "[2/5] RTX 5060(sm_120) Detectron2 + SAM2"
POSTCLASS_CUDA_HOME="$(conda run -n phmr_postclass_omen5060 bash -lc 'printf %s "$CONDA_PREFIX"')"
conda run -n phmr_postclass_omen5060 env \
  CUDA_HOME="$POSTCLASS_CUDA_HOME" TORCH_CUDA_ARCH_LIST=12.0 \
  python -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/detectron2.git@247691a1c292654867deb35cabb27f18d75a1363"
conda run -n phmr_postclass_omen5060 env SAM2_BUILD_CUDA=0 \
  python -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"

echo "[3/5] 독립 GMR 환경"
ensure_env gmr_postclass_omen5060
conda run -n gmr_postclass_omen5060 python -m pip install \
  torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
conda run -n gmr_postclass_omen5060 python -m pip install -e "$LAB_ROOT/repos/GMR"

echo "[4/5] 독립 MjLab/PPO 환경"
ensure_env mjlab_postclass_omen5060
conda run -n mjlab_postclass_omen5060 python -m pip install \
  torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
conda run -n mjlab_postclass_omen5060 python -m pip install \
  -e "$LAB_ROOT/repos/unitree_rl_mjlab" scipy tensorboard wandb \
  "warp-lang==1.12.0" "mujoco==3.5.0" "mujoco-warp==3.5.0"

echo "[5/5] PromptHMR 공개 checkpoint"
bash "$LAB_ROOT/postclass_full_pipeline/fetch_weights.sh"

echo
echo "별도 환경 설치 완료. 다음 단계:"
echo "1) README에 따라 SMPL/SMPL-X ZIP을 공식 사이트에서 받기"
echo "2) bash postclass_full_pipeline/install_body_models.sh <SMPL-X.zip> <SMPL.zip>"
echo "3) bash postclass_full_pipeline/verify.sh"


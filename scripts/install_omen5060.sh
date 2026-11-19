#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--execute" ]]; then
  echo "HP OMEN 16 / RTX 5060용 환경 3개를 만들고 대용량 weight를 내려받습니다."
  echo "실행: bash scripts/install_omen5060.sh --execute"
  exit 2
fi

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHMR="$LAB_ROOT/repos/PromptHMR"
for COMMAND in conda git wget unzip ffmpeg nvidia-smi; do
  if ! command -v "$COMMAND" >/dev/null 2>&1; then
    echo "필수 명령을 찾을 수 없습니다: $COMMAND" >&2
    echo "README의 '노트북 준비'를 먼저 실행하세요." >&2
    exit 1
  fi
done
eval "$(conda shell.bash hook)"

ensure_conda_env() {
  local env_name="$1"
  local python_version="$2"
  if conda run -n "$env_name" python -c "import sys" >/dev/null 2>&1; then
    echo "[resume] 기존 Conda 환경 재사용: $env_name"
  else
    conda create -n "$env_name" "python=$python_version" -y
  fi
}

echo "[1/4] PromptHMR: Python 3.11 + PyTorch 2.7.1/cu128"
ensure_conda_env phmr_omen5060 3.11
conda install -n phmr_omen5060 -c nvidia/label/cuda-12.8.1 cuda-toolkit -y
conda run -n phmr_omen5060 python -m pip install --upgrade "pip<26" setuptools wheel ninja
conda run -n phmr_omen5060 python -m pip install \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
conda run -n phmr_omen5060 python -m pip install \
  torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
conda run -n phmr_omen5060 python -m pip install "numpy<2" -r "$PHMR/requirements.txt"
echo "[1/4] chumpy (legacy setup.py: build isolation disabled)"
conda run -n phmr_omen5060 python -m pip install --no-build-isolation \
  "git+https://github.com/Arthur151/chumpy.git@ec5a20f3adc1a105c7a907cfed9a33d7dccbe95f"
conda run -n phmr_omen5060 python -m pip install xformers==0.0.31.post1 \
  --index-url https://download.pytorch.org/whl/cu128 --no-deps

echo "[2/4] RTX 5060(sm_120)용 Detectron2 source build + SAM2"
CUDA_HOME="$(conda run -n phmr_omen5060 bash -lc 'printf %s "$CONDA_PREFIX"')"
export CUDA_HOME TORCH_CUDA_ARCH_LIST="12.0"
conda run -n phmr_omen5060 env CUDA_HOME="$CUDA_HOME" TORCH_CUDA_ARCH_LIST=12.0 \
  python -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/detectron2.git@247691a1c292654867deb35cabb27f18d75a1363"
conda run -n phmr_omen5060 env SAM2_BUILD_CUDA=0 \
  python -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"

echo "[3/4] PromptHMR network weights (시간과 디스크가 많이 필요합니다)"
pushd "$PHMR" >/dev/null
if [[ -f data/pretrain/phmr/checkpoint.ckpt \
   && -f data/pretrain/phmr_vid/prhmr_release_002.ckpt \
   && -f data/pretrain/sam2_ckpts/sam2_hiera_tiny.pt \
   && -f data/pretrain/vitpose-h-coco_25.pth ]]; then
  echo "[resume] PromptHMR 핵심 weight가 이미 있어 다운로드를 건너뜁니다."
else
  conda run --no-capture-output -n phmr_omen5060 bash scripts/fetch_data.sh
fi
find data -type d -name wheels -prune -exec rm -rf {} +
rm -f data/pretrain/sam_vit_h_4b8939.pth
popd >/dev/null

echo "[4/4] GMR와 Unitree RL MjLab"
for ENV_NAME in gmr_omen5060 unitree_mjlab_omen5060; do
  ensure_conda_env "$ENV_NAME" 3.11
  conda run -n "$ENV_NAME" python -m pip install --upgrade "pip<26" setuptools wheel
  conda run -n "$ENV_NAME" python -m pip install \
    torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
done
conda run -n gmr_omen5060 python -m pip install -e "$LAB_ROOT/repos/GMR"
conda run -n unitree_mjlab_omen5060 python -m pip install -e "$LAB_ROOT/repos/unitree_rl_mjlab"

echo
echo "완료: phmr_omen5060 / gmr_omen5060 / unitree_mjlab_omen5060"
echo "이제 README의 SMPL/SMPL-X 다운로드를 진행하세요."

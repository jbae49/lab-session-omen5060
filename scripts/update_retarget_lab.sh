#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="unitree_mjlab_omen5060"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda 명령을 찾을 수 없습니다." >&2
  exit 1
fi
if ! conda run -n "$ENV_NAME" python -c "import sys" >/dev/null 2>&1; then
  echo "$ENV_NAME 환경이 없습니다." >&2
  echo "처음 설치라면 실행하세요: bash scripts/install_retarget_lab.sh --execute" >&2
  exit 1
fi

echo "수업용 MjLab 환경의 호환 패키지만 업데이트합니다."
conda run --no-capture-output -n "$ENV_NAME" python -m pip install --upgrade \
  scipy "mujoco==3.5.0" "mujoco-warp==3.5.0"

echo "설치 결과를 검사합니다."
conda run --no-capture-output -n "$ENV_NAME" \
  python "$LAB_ROOT/scripts/doctor.py" --stage mjlab

echo
echo "업데이트 완료. 이제 실행하세요:"
echo "  bash scripts/run_retarget_lab.sh"

#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for ENV_NAME in phmr_postclass_omen5060 gmr_postclass_omen5060 mjlab_postclass_omen5060; do
  if ! conda run -n "$ENV_NAME" python -c "import sys" >/dev/null 2>&1; then
    echo "[MISSING] Conda environment: $ENV_NAME" >&2
    exit 1
  fi
  echo "[OK] Conda environment: $ENV_NAME"
done

conda run --no-capture-output -n phmr_postclass_omen5060 \
  python "$LAB_ROOT/scripts/doctor.py" --stage prompthmr
conda run --no-capture-output -n gmr_postclass_omen5060 \
  python "$LAB_ROOT/scripts/doctor.py" --stage gmr
conda run --no-capture-output -n mjlab_postclass_omen5060 \
  python "$LAB_ROOT/scripts/doctor.py" --stage mjlab

echo "[OK] 수업 후 전체 파이프라인 환경 검사 완료"


#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHMR="$LAB_ROOT/repos/PromptHMR"

if ! conda run -n phmr_postclass_omen5060 python -c "import gdown" >/dev/null 2>&1; then
  echo "phmr_postclass_omen5060 환경이 없습니다. install.sh를 먼저 실행하세요." >&2
  exit 1
fi

if [[ -f "$PHMR/data/pretrain/phmr/checkpoint.ckpt" \
   && -f "$PHMR/data/pretrain/phmr_vid/prhmr_release_002.ckpt" \
   && -f "$PHMR/data/pretrain/sam2_ckpts/sam2_hiera_tiny.pt" \
   && -f "$PHMR/data/pretrain/vitpose-h-coco_25.pth" \
   && -f "$PHMR/data/body_models/smplx2smpl.pkl" \
   && -f "$PHMR/data/body_models/smplx2smpl_joints.npy" ]]; then
  echo "[OK] 핵심 PromptHMR checkpoint가 이미 있습니다."
  exit 0
fi

echo "PromptHMR 공식 fetch_data.sh를 실행합니다. Google Drive quota에 따라 재시도가 필요할 수 있습니다."
(
  cd "$PHMR"
  conda run --no-capture-output -n phmr_postclass_omen5060 bash scripts/fetch_data.sh
)

find "$PHMR/data" -type d -name wheels -prune -exec rm -rf {} +
rm -f "$PHMR/data/pretrain/sam_vit_h_4b8939.pth"
echo "[OK] checkpoint 다운로드 단계 완료"


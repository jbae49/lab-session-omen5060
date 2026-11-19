#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /path/to/models_smplx_v1_1.zip /path/to/SMPL_python_v.1.1.0.zip" >&2
  exit 2
fi

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMPLX_ZIP="$(realpath "$1")"
SMPL_ZIP="$(realpath "$2")"
for ARCHIVE in "$SMPLX_ZIP" "$SMPL_ZIP"; do
  if [[ ! -f "$ARCHIVE" ]]; then
    echo "ZIP을 찾을 수 없습니다: $ARCHIVE" >&2
    exit 1
  fi
done

MODEL_TMP="$(mktemp -d)"
trap 'rm -rf "$MODEL_TMP"' EXIT
unzip -q "$SMPLX_ZIP" -d "$MODEL_TMP/smplx"
unzip -q "$SMPL_ZIP" -d "$MODEL_TMP/smpl"

find_one() {
  local root="$1"
  local pattern="$2"
  local found
  mapfile -t found < <(find "$root" -type f -iname "$pattern")
  if [[ ${#found[@]} -ne 1 ]]; then
    echo "'$pattern' 파일을 정확히 하나 찾아야 하지만 ${#found[@]}개를 찾았습니다." >&2
    printf '  %s\n' "${found[@]}" >&2
    exit 1
  fi
  printf '%s\n' "${found[0]}"
}

PHMR_MODELS="$LAB_ROOT/repos/PromptHMR/data/body_models"
GMR_MODELS="$LAB_ROOT/repos/GMR/assets/body_models/smplx"
mkdir -p "$PHMR_MODELS/smplx" "$PHMR_MODELS/smpl" "$GMR_MODELS"

for GENDER in NEUTRAL MALE FEMALE; do
  SOURCE="$(find_one "$MODEL_TMP/smplx" "SMPLX_${GENDER}.npz")"
  install -m 0644 "$SOURCE" "$PHMR_MODELS/smplx/SMPLX_${GENDER}.npz"
  install -m 0644 "$SOURCE" "$GMR_MODELS/SMPLX_${GENDER}.npz"
done

install -m 0644 "$(find_one "$MODEL_TMP/smpl" 'basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl')" \
  "$PHMR_MODELS/smpl/SMPL_NEUTRAL.pkl"
install -m 0644 "$(find_one "$MODEL_TMP/smpl" 'basicmodel_m_lbs_10_207_0_v1.1.0.pkl')" \
  "$PHMR_MODELS/smpl/SMPL_MALE.pkl"
install -m 0644 "$(find_one "$MODEL_TMP/smpl" 'basicmodel_f_lbs_10_207_0_v1.1.0.pkl')" \
  "$PHMR_MODELS/smpl/SMPL_FEMALE.pkl"

echo "body model 복사 완료. PromptHMR 변환 보조 파일은 fetch_weights.sh에서 준비됩니다."


#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$LAB_ROOT/repos/PromptHMR/data/body_models/smplx"
DESTINATION="$LAB_ROOT/repos/GMR/assets/body_models/smplx"

mkdir -p "$DESTINATION"
for gender in NEUTRAL MALE FEMALE; do
  filename="SMPLX_${gender}.npz"
  if [[ ! -f "$SOURCE/$filename" ]]; then
    echo "Missing $SOURCE/$filename" >&2
    echo "Official SMPL-X download: https://smpl-x.is.tue.mpg.de/download.php" >&2
    echo "First run: cd repos/PromptHMR && bash scripts/fetch_smplx.sh" >&2
    exit 1
  fi
  cp -f "$SOURCE/$filename" "$DESTINATION/$filename"
  echo "Copied $filename"
done

echo "GMR now has independent model files (not symbolic links)."

#!/usr/bin/env bash
# Backward-compatible entry point. The actual installer is RTX 5060 specific.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/install_omen5060.sh" "$@"

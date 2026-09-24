#!/usr/bin/env bash
set -euo pipefail

HANDOFF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GSPAT_SOURCE="${1:?usage: revert_warp_emit.sh /path/to/gsplat}"
test -f "$GSPAT_SOURCE/cuda/csrc/IntersectTile.cu"
(cd "$GSPAT_SOURCE" && patch -R -p1 --forward < "$HANDOFF_DIR/warp_emit.patch")

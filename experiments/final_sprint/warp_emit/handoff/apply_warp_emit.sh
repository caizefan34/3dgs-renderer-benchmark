#!/usr/bin/env bash
set -euo pipefail

# Apply only to the gsplat source tree audited at commit 02375033388d4348376b6b607ab85f551e498a77.
HANDOFF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GSPAT_SOURCE="${1:?usage: apply_warp_emit.sh /path/to/gsplat}"
test -f "$GSPAT_SOURCE/cuda/csrc/IntersectTile.cu"
(cd "$GSPAT_SOURCE" && patch -p1 --forward < "$HANDOFF_DIR/warp_emit.patch")

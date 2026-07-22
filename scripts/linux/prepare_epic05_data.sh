#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
STATE_ROOT="${STATE_ROOT:-/mnt/workspace/codex-3dgs-epic05}"
PYTHON="${PYTHON:-$HOME/miniforge3/envs/gsplat/bin/python}"
DOWNLOADS="$STATE_ROOT/downloads"
DATA_ROOT="${DATA_ROOT:-$STATE_ROOT/datasets}"

wait_for_size() {
  local path="$1"
  local expected="$2"
  while [[ ! -f "$path" || "$(stat -c %s "$path")" -ne "$expected" ]]; do
    current=0
    [[ -f "$path" ]] && current="$(stat -c %s "$path")"
    echo "waiting for $(basename "$path"): $current / $expected bytes"
    sleep 30
  done
}

while [[ ! -x "$PYTHON" ]] || \
  ! "$PYTHON" -c "import google_crc32c, PIL, remotezip" >/dev/null 2>&1; do
  echo "waiting for benchmark environment: $PYTHON"
  sleep 30
done

mkdir -p "$DATA_ROOT" "$ROOT/datasets"
for directory in raw processed downloads candidates; do
  mkdir -p "$DATA_ROOT/$directory"
  if [[ ! -e "$ROOT/datasets/$directory" ]]; then
    ln -s "$DATA_ROOT/$directory" "$ROOT/datasets/$directory"
  fi
done

wait_for_size "$DOWNLOADS/mipnerf360-garden.zip" 2986730894
wait_for_size "$DOWNLOADS/mipnerf360-bicycle.zip" 2476191765
wait_for_size "$DOWNLOADS/mipnerf360-bonsai.zip" 1395703750
wait_for_size "$DOWNLOADS/tandt_db.zip" 682628995

for scene in garden bicycle bonsai; do
  "$PYTHON" "$ROOT/src/scripts/prepare_datasets.py" mipnerf360 \
    --data-root "$DATA_ROOT" --scene "$scene" \
    --archive "$DOWNLOADS/mipnerf360-$scene.zip"
done
for scene in truck train; do
  "$PYTHON" "$ROOT/src/scripts/prepare_datasets.py" tanks_and_temples \
    --data-root "$DATA_ROOT" --scene "$scene" \
    --archive "$DOWNLOADS/tandt_db.zip"
done

for case_id in \
  small-garden-1080p medium-truck-1080p medium-train-1080p \
  large-bicycle-1080p large-bonsai-1080p; do
  case_path="$ROOT/$($PYTHON - "$ROOT/benchmark/suite.json" "$case_id" <<'PY'
import json
import sys

suite = json.load(open(sys.argv[1], encoding="utf-8"))
print(next(row["scene_path"] for row in suite["cases"] if row["case_id"] == sys.argv[2]))
PY
)"
  if [[ ! -f "$case_path" ]]; then
    "$PYTHON" "$ROOT/src/scripts/prepare_suite_case.py" "$case_id"
  fi
done

echo "EPIC-05 canonical data are ready under $DATA_ROOT"

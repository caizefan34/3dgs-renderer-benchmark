#!/bin/bash
# collect_r3_results.sh - wait for completion and gather all results
REPO="$(ls -d /home/*/3dgs-renderer-benchmark 2>/dev/null | head -1)"
R3DIR="$REPO/results/reference_v1/r3"

echo "=== All R3 output directories ==="
for D in "$R3DIR"/*/; do
  echo "--- Directory: $D ---"
  ls -la "$D" 2>/dev/null | head -20
done

echo "=== Aggregate JSON files ==="
find "$R3DIR" -name "*.json" 2>/dev/null | sort | while read f; do
  echo "=== $f ==="
  cat "$f" 2>/dev/null | head -50
done

echo "=== NPZ archives ==="
find "$R3DIR" -name "*.npz" 2>/dev/null | while read f; do
  echo "Archive: $f  ($(stat -c '%s' "$f") bytes)"
done

echo "=== PNG figures ==="
find "$R3DIR" -name "*.png" 2>/dev/null | while read f; do
  echo "Figure: $f"
done

echo "=== Latest runner log ==="
tail -30 "$R3DIR/5000/runner.log" 2>/dev/null

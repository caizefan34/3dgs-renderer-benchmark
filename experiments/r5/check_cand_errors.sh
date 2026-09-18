#!/bin/bash
echo "=== Full candidate logs (searching for errors) ==="
for s in 1 2; do
  for scene in train truck; do
    log="/mnt/storage_pool/liaoyuanjun/r5_a_logs/${scene}_seed${s}_candidate.log"
    echo "--- $scene seed=$s candidate (grep error/traceback) ---"
    grep -i "error\|traceback\|exception\|OOM\|memory\|killed\|RuntimeError" "$log" 2>/dev/null | head -5
    echo ""
    echo "--- $scene seed=$s candidate (wc -l) ---"
    wc -l "$log" 2>/dev/null
    echo ""
  done
done

echo "=== Check if stderr captured ==="
for s in 1 2; do
  for scene in train truck; do
    log="/mnt/storage_pool/liaoyuanjun/r5_a_logs/${scene}_seed${s}_candidate.log"
    echo "--- $scene seed=$s: last 10 lines ---"
    tail -10 "$log" 2>/dev/null
    echo ""
  done
done

#!/bin/bash
# FINAL30K gates runner: idempotent, runs all remaining roles sequentially.
export CUDA_VISIBLE_DEVICES=2
PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
OUT=/mnt/storage_pool/liaoyuanjun/final30k_gates
mkdir -p "$OUT"

for scene in room bicycle garden; do
  for v in old old_b new_abs0 new_abs1; do
    f="$OUT/c0_${v}_${scene}.pt"
    if [ ! -f "$f" ]; then
      echo "=== c0 $v $scene ==="
      $PY /tmp/final30k_gates.py c0 "$scene" "$OUT" --variant "$v" > "$OUT/log_c0_${v}_${scene}.txt" 2>&1 || echo "FAIL c0 $v $scene"
    else
      echo "skip c0 $v $scene (exists)"
    fi
  done
  f="$OUT/capture_${scene}.pt"
  if [ ! -f "$f" ]; then
    echo "=== capture $scene ==="
    $PY /tmp/final30k_gates.py capture "$scene" "$OUT" > "$OUT/log_capture_${scene}.txt" 2>&1 || echo "FAIL capture $scene"
  else
    echo "skip capture $scene (exists)"
  fi
  f="$OUT/b1a_d2_${scene}.pt"
  if [ ! -f "$f" ]; then
    echo "=== b1a_d2 $scene ==="
    $PY /tmp/final30k_gates.py b1a_d2 "$scene" "$OUT" > "$OUT/log_b1a_d2_${scene}.txt" 2>&1 || echo "FAIL b1a_d2 $scene"
  else
    echo "skip b1a_d2 $scene (exists)"
  fi
  f="$OUT/b1a_d1_${scene}.pt"
  if [ ! -f "$f" ]; then
    echo "=== b1a_d1 $scene ==="
    $PY /tmp/final30k_gates.py b1a_d1 "$scene" "$OUT" > "$OUT/log_b1a_d1_${scene}.txt" 2>&1 || echo "FAIL b1a_d1 $scene"
  else
    echo "skip b1a_d1 $scene (exists)"
  fi
done

for scene in room bicycle garden; do
  echo "=== compare $scene ==="
  $PY /tmp/final30k_gates.py compare "$scene" "$OUT" > "$OUT/log_compare_${scene}.txt" 2>&1 || echo "FAIL compare $scene"
done
echo ALL_DONE

#!/usr/bin/env bash
Q=/mnt/storage_pool/liaoyuanjun/C0_QUALITY_STABILITY0_MX
echo "=== top-level ==="; ls -la "$Q"
echo "=== required subdirs (goal 38) ==="
for d in raw logs commands checkpoints gpu_snapshots scripts; do
  if [ -d "$Q/$d" ]; then echo "OK  $d ($(ls "$Q/$d" | wc -l) entries)"; else echo "MISSING $d"; fi
done
echo "=== artifacts present ==="
for f in QUALITY_SOURCE_IDENTITY.json QUALITY_PROTOCOL_IDENTITY.json QUALITY_SEED42_REUSE.json QUALITY_RUN_PLAN.json QUALITY_CONTAMINATION.json QUALITY_FROZEN_GATES.json; do
  [ -f "$Q/$f" ] && echo "OK  $f" || echo "MISSING $f"
done
echo "=== seed42 reuse checkpoints ==="; ls "$Q/checkpoints" | grep s42 | wc -l

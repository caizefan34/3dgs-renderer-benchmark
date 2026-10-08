#!/bin/bash
set -u
echo "=== DPC .so ==="
DPCSO=/root/.cache/torch_extensions/py310_cu128/dpc0_gaussian_render_inference_scene_cuda/dpc0_gaussian_render_inference_scene_cuda.so
if [ -f "$DPCSO" ]; then
  ls -la "$DPCSO"
  echo "actual:   $(sha256sum "$DPCSO" | cut -d" " -f1)"
  echo "expected: 8742df90c95f6fa1464954dc91ec0f4510b49a39363f9321360c24677f93c16e"
else
  echo "MISSING DPC .so"
fi
echo
echo "=== checkpoints ==="
declare -A EXP
EXP[/root/epic05_n1/runs/n1snap_c0f_s42_room/ckpt_10000.pt]=9aa00f0ddf28b6c097e3dc8f41dbb107cd9bf175750503dc330aa1d6ab579fc4
EXP[/root/epic05_n1/runs/n1snap_c0f_s42_room/ckpt_15000.pt]=b276b338f5a444d8ea1be390d21fd3e67679b7d862652f0babd131db9434e5dd
EXP[/root/epic05_n1/runs/n1snap_c0f_s42_bicycle/ckpt_10000.pt]=26bf0b959f10f0ce526c496d0dc7ca8e62b0ba93efc83d58b7f9440d58c2f4e9
EXP[/root/epic05_n1/runs/n1snap_c0f_s42_bicycle/ckpt_15000.pt]=273971d3867c1d2a08b7246803265539d74756b8eca4e3f01c6a8d149b7fd6f7
ALL_OK=1
for p in "${!EXP[@]}"; do
  if [ -f "$p" ]; then
    actual=$(sha256sum "$p" | cut -d" " -f1)
    if [ "$actual" = "${EXP[$p]}" ]; then st=OK; else st=MISMATCH; ALL_OK=0; fi
    echo "$st $p -> $actual"
  else
    echo "MISSING $p"; ALL_OK=0
  fi
done
echo "CHECKPOINTS_ALL_OK=$ALL_OK"
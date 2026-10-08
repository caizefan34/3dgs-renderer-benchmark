#!/bin/bash
R=/root/3dgs-renderer-benchmark
GM=$R/baseline/reference_v1/gaussian_model.py
echo "=== getters / activations ==="
grep -n "def get_xyz\|def get_rotation\|def get_scaling\|def get_opacity\|def get_features\|scaling_activation\|opacity_activation\|def capture\|_features\b" $GM | head -30
echo
echo "=== getter bodies ==="
for fn in get_xyz get_rotation get_scaling get_opacity get_features; do
  N=$(grep -n "def $fn" $GM | head -1 | cut -d: -f1)
  if [ -n "$N" ]; then sed -n "${N},$((N+6))p" $GM; echo "---"; fi
done
echo "=== training_setup param groups ==="
N=$(grep -n "def training_setup" $GM | head -1 | cut -d: -f1)
sed -n "${N},$((N+45))p" $GM

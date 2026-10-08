#!/bin/bash
set -u
G=/root/3dgs-renderer-benchmark/baseline/reference_v1
echo "=== config.py full ==="
cat $G/config.py
echo "=== gaussian_model key methods ==="
grep -n 'def \|_features\|active_sh_degree\|sh_progress' $G/gaussian_model.py | head -50
echo "=== capture ==="
sed -n "$(grep -n 'def capture' $G/gaussian_model.py | head -1 | cut -d: -f1),+20p" $G/gaussian_model.py
echo "=== training_setup optimizer groups ==="
sed -n "$(grep -n 'def training_setup' $G/gaussian_model.py | head -1 | cut -d: -f1),+60p" $G/gaussian_model.py
echo "=== dataset.py GTDataset get_item / resolution ==="
grep -n 'def \|resolution\|1080p\|downscale' $G/dataset.py | head -40

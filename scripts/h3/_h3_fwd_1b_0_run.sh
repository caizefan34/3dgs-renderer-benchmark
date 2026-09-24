#!/bin/bash
set -e
export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:/home/liaoyuanjun/.local/bin:$PATH
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun/higs-13scene-env
export CUDA_VISIBLE_DEVICES=4

PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
SOURCE=/tmp/higs_h3_fwd_1a_source
CORE_SO=/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so
OUT_DIR=/tmp/higs_h3_fwd_1b_0

echo "=== H3-FWD-1B-0: Macro Raster Break-even Oracle ==="

$PY /tmp/h3_fwd_1b_0_break_even_oracle.py \
    --out-dir "$OUT_DIR" \
    --source "$SOURCE" \
    --core-so "$CORE_SO" \
    --gpu 4 \
    --max-long-side 2048 \
    --warmup 20 \
    --measure 100 \
    --reps 5 \
    --seed 4200

echo ""
echo "=== Results ==="
echo "--- b2_f5_timing.csv ---"
cat "$OUT_DIR/b2_f5_timing.csv"
echo ""
echo "--- break_even_targets.json ---"
cat "$OUT_DIR/break_even_targets.json"
echo ""
echo "--- b2_f5_workload.json (summary) ---"
python3 -c "
import json
d = json.load(open('$OUT_DIR/b2_f5_workload.json'))
for s in ['room','bicycle','garden']:
    w = d[s]
    print(f'{s}: {w[\"n_fine_tile_gaussian_pairs\"]} pairs, {w[\"n_active_tiles\"]} tiles, per-pixel mean={w[\"per_pixel_evaluations\"][\"mean\"]:.1f} p90={w[\"per_pixel_evaluations\"][\"p90\"]:.1f}')
" 2>/dev/null
echo ""
echo "--- existing_higs_raster_oracle.json ---"
cat "$OUT_DIR/existing_higs_raster_oracle.json"
echo ""
echo "--- opportunity_analysis.json (decision) ---"
python3 -c "
import json
d = json.load(open('$OUT_DIR/opportunity_analysis.json'))
print('Decision:', d['decision']['gate'])
print('Avg F4 penalty % of F5:', d['decision']['avg_f4_penalty_pct_of_f5'])
" 2>/dev/null

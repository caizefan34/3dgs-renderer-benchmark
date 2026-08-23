"""Cross-cohort analysis: merge synthetic and official results, generate summary."""
import json, os
from pathlib import Path

REPO_ROOT = Path(os.path.expanduser('~/3dgs-renderer-benchmark'))
RESULTS = REPO_ROOT / 'results' / 'epic05'

# 1. Load synthetic results (ablation)
synthetic = {}
for fname in ['ablation_50k_1080p.json', 'ablation_200k_1080p.json', 'ablation_400k_1080p.json']:
    fp = RESULTS / 'raw' / fname
    if not fp.exists():
        continue
    with open(fp) as f:
        data = json.load(f)
    # structure: dict with scene_X keys
    for k, v in data.items():
        if isinstance(v, dict):
            for ts_key, ts_val in v.items():
                if isinstance(ts_val, dict):
                    t16 = ts_val.get('tile16') or ts_val.get('tile16', {})
                    t32 = ts_val.get('tile32') or ts_val.get('tile32', {})
                    t8 = ts_val.get('tile8') or ts_val.get('tile8', {})
                    if isinstance(t16, dict) and 'mean_ms' in t16:
                        synthetic[f'{k}_{ts_key}'] = {
                            'num_gaussians': v.get('num_gaussians') or ts_val.get('num_gaussians', 0),
                            'tile8_ms': t8.get('mean_ms') if isinstance(t8, dict) else None,
                            'tile16_ms': t16['mean_ms'],
                            'tile32_ms': t32.get('mean_ms') if isinstance(t32, dict) else None,
                        }

print('Synthetic entries:', len(synthetic))
for k, v in sorted(synthetic.items()):
    print(f'  {k}: {v["num_gaussians"]} GS, t16={v["tile16_ms"]:.2f}ms, t32={v["tile32_ms"]}')

# 2. Load official aggregated (data['rows'] is the list)
with open(RESULTS / 'official' / 'aggregated' / 'official_aggregated.json') as f:
    data = json.load(f)
official = data['rows']

print('\nOfficial entries:', len(official))
official_clean = {}
for row in official:
    sid = row['scene_id']
    protocol = row.get('protocol', {})
    repeats = protocol.get('repeats', 0)
    # Skip single-repeat entries (prefer 3-repeat)
    if repeats < 3:
        print(f'  SKIP {sid} (repeats={repeats})')
        continue
    t16 = row.get('tile16', {})
    t32 = row.get('tile32', {})
    t8 = row.get('tile8', {})
    if isinstance(t16, dict) and 'mean_ms' in t16:
        official_clean[sid] = {
            'num_gaussians': row['num_gaussians'],
            'tile8_ms': t8.get('mean_ms') if isinstance(t8, dict) else None,
            'tile16_ms': t16['mean_ms'],
            'tile32_ms': t32.get('mean_ms') if isinstance(t32, dict) else None,
            'vram_tile8': t8.get('peak_vram_mb') if isinstance(t8, dict) else None,
            'vram_tile16': t16.get('peak_vram_mb'),
            'vram_tile32': t32.get('peak_vram_mb') if isinstance(t32, dict) else None,
        }
        t32_s = f'{t32["mean_ms"]:.2f}' if isinstance(t32, dict) and 'mean_ms' in t32 else 'N/A'
        print(f'  {sid}: {row["num_gaussians"]} GS, t16={t16["mean_ms"]:.2f}ms, t32={t32_s}')

# 3. Cross-cohort comparison
print('\n' + '='*70)
print('CROSS-COHORT TILE_SIZE SPEEDUP COMPARISON')
print('='*70)
print(f'{"Dataset":<14} {"GS Count":<12} {"t8(ms)":<10} {"t16(ms)":<10} {"t32(ms)":<10} {"t32/t16":<10}')
print('-'*70)

all_data = {}
for k, v in synthetic.items():
    all_data[f'SYN-{k}'] = v
for k, v in official_clean.items():
    all_data[f'OFF-{k}'] = v

for label, v in sorted(all_data.items()):
    t8 = v.get('tile8_ms')
    t16 = v['tile16_ms']
    t32 = v.get('tile32_ms')
    t8_s = f'{t8:.2f}' if t8 else 'N/A'
    t32_s = f'{t32:.2f}' if t32 else 'N/A'
    speedup = f'{t16/t32:.4f}x' if t32 else 'N/A'
    print(f'{label:<14} {v["num_gaussians"]:<12,} {t8_s:<10} {t16:<10.2f} {t32_s:<10} {speedup:<10}')

# 4. Analysis: tile32 speedup vs gaussian count
print('\n' + '='*70)
print('SPEEDUP vs GAUSSIAN COUNT (tile32 / tile16)')
print('='*70)
for label, v in sorted(all_data.items()):
    if v.get('tile32_ms'):
        speedup = v['tile16_ms'] / v['tile32_ms']
        print(f'  {label:<14} GS={v["num_gaussians"]:<10,} t16={v["tile16_ms"]:<8.2f}ms t32={v["tile32_ms"]:<8.2f}ms speedup={speedup:<8.4f}x {">" if speedup > 1 else "<"} 1.0')

# 5. Key finding summary
print('\n' + '='*70)
print('KEY FINDINGS')
print('='*70)
# Real scenes: which tile_size is fastest?
for label, v in sorted(official_clean.items()):
    t8 = v.get('tile8_ms')
    t16 = v['tile16_ms']
    t32 = v.get('tile32_ms')
    best = min([(k, val) for k, val in [('t8', t8), ('t16', t16), ('t32', t32)] if val], key=lambda x: x[1])
    print(f'  OFF-{label}: Best tile_size={best[0]} @ {best[1]:.2f}ms ({v["num_gaussians"]:,} GS)')

# Save summary
out_path = RESULTS / 'official' / 'statistics' / 'cross_cohort_summary.json'
os.makedirs(out_path.parent, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump({'synthetic': synthetic, 'official': official_clean}, f, indent=2, default=str)
print(f'\nCross-cohort summary saved: {out_path}')

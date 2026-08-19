#!/usr/bin/env python3
"""Deep-dive into 4K bicycle results."""
import json
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\36570\3dgs-renderer-benchmark")

# Load 4K file
fpath = REPO_ROOT / "results" / "epic05" / "phase4" / "bicycle_rerun_20260819_021657.json"
with open(fpath) as f:
    data = json.load(f)

bike = data['scenes']['bicycle']['tile_results']

print("4K Bicycle Results Analysis")
print("=" * 60)

for tk in ["tile16", "tile32"]:
    tr = bike[tk]
    print(f"\n--- {tk} ---")
    print(f"  Overall: mean={tr['mean_ms']:.2f}ms, stable_mean={tr['stable_mean_ms']:.2f}ms")
    print(f"  median={tr['median_ms']:.2f}ms, p99={tr['p99_ms']:.2f}ms")
    print(f"  outliers={tr['outlier_analysis']['count_severe_outliers']}")
    print(f"  repeats:")
    for rep in tr['repeat_statistics']:
        clean = tr['outlier_analysis']['count_severe_outliers'] == 0 or True
        print(f"    rep {rep['repeat']}: mean={rep['mean_ms']:.2f}ms, median={rep['median_ms']:.2f}ms, max={rep['max_ms']:.2f}ms")
    # Extract clean frames
    times = tr['frame_times_ms']
    clean_times = [t for t in times if t <= 100]
    outlier_times = [t for t in times if t > 100]
    print(f"  Clean frames: {len(clean_times)}, outlier frames: {len(outlier_times)}")
    if clean_times:
        import numpy as np
        ct = np.array(clean_times)
        print(f"  Clean stats: mean={ct.mean():.2f}ms, median={np.median(ct):.2f}ms, p99={np.percentile(ct, 99):.2f}ms")
    if outlier_times:
        print(f"  Outlier stats: count={len(outlier_times)}, max={max(outlier_times):.1f}ms, min={min(outlier_times):.1f}ms")

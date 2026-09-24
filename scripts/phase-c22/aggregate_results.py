import json, numpy as np, sys

files = sys.argv[1:]

all_wr = []
all_util = []
all_n12 = []
cam_details = []

for f in files:
    d = json.load(open(f))
    for rec in d['camera_records']:
        wr = rec['frame_work_reduction_bound']['potential_work_reduction_fraction']
        all_wr.append(wr)
        all_util.append(rec['useful_prefix_ratio']['mean'])
        all_n12.append(rec['n12_active_pixel_fraction_near_end']['mean'])
        total = rec['frame_work_reduction_bound']['total_rasterizer_gaussian_steps']
        skipped = rec['frame_work_reduction_bound']['potentially_skipped_steps']
        cam_details.append({
            'cam': rec['camera'],
            'work_reduction': wr,
            'useful_ratio_mean': rec['useful_prefix_ratio']['mean'],
            'total_steps': total,
            'skipped': skipped,
        })

wr = np.array(all_wr)
util = np.array(all_util)
n12 = np.array(all_n12)

total_all_steps = sum(c['total_steps'] for c in cam_details)
total_all_skipped = sum(c['skipped'] for c in cam_details)

print("=== OVERALL ===")
print(f"Cameras measured: {len(cam_details)}")
print()
print("POTENTIAL WORK REDUCTION (skippable fraction):")
for label, fn in [("Mean", np.mean), ("P50", lambda x: np.percentile(x,50)),
                   ("P90", lambda x: np.percentile(x,90)), ("P95", lambda x: np.percentile(x,95)),
                   ("P99", lambda x: np.percentile(x,99)), ("Min", np.min), ("Max", np.max)]:
    print(f"  {label}: {fn(wr):.4f}")

print()
print("USEFUL PREFIX RATIO (fraction of sorted range actually used):")
for label, fn in [("Mean", np.mean), ("P50", lambda x: np.percentile(x,50)),
                   ("P90", lambda x: np.percentile(x,90)), ("Min", np.min), ("Max", np.max)]:
    print(f"  {label}: {fn(util):.4f}")

print()
print("N12 active-pixel fraction near end:")
for label, fn in [("Mean", np.mean), ("P50", lambda x: np.percentile(x,50)),
                   ("P90", lambda x: np.percentile(x,90)), ("Min", np.min), ("Max", np.max)]:
    print(f"  {label}: {fn(n12):.4f}")

print()
print(f"Total rasterizer steps: {total_all_steps:,}")
print(f"Total potentially skipped: {total_all_skipped:,}")
print(f"Aggregate work reduction: {total_all_skipped / max(total_all_steps, 1):.4f}")

print()
print("CDF: cameras with work_reduction <= threshold:")
for th in [0.10, 0.20, 0.30, 0.50, 0.70, 0.80]:
    cnt = sum(1 for w in all_wr if w <= th)
    print(f"  <= {th:.2f}: {cnt}/{len(all_wr)} = {100*cnt/len(all_wr):.1f}%")

print()
print("Pixels with useful_prefix_ratio <= threshold (first camera):")
d0 = json.load(open(files[0]))
rec0 = d0['camera_records'][0]
for k,v in rec0['useful_prefix_ratio_cdf'].items():
    print(f"  {k}: {v:.4f}")

print()
# Also print per-file summary
for f in files:
    d = json.load(open(f))
    group = f.split("_")[-1].replace(".json","")
    grp_wrs = [r['frame_work_reduction_bound']['potential_work_reduction_fraction'] for r in d['camera_records']]
    print(f"{group}: n={len(grp_wrs)} mean_wr={np.mean(grp_wrs):.4f} range=[{min(grp_wrs):.4f},{max(grp_wrs):.4f}]")

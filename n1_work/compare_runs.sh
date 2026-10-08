#!/bin/bash
PY=/root/miniforge3/envs/gsplat/bin/python
echo "=== canonical p12a room run ==="
cat /root/epic05_p12/runs/p12a/p12a_c0f_s42_room/training_results.json
echo "=== canonical p12a bicycle run ==="
cat /root/epic05_p12/runs/p12a/p12a_c0f_s42_bicycle/training_results.json
echo "=== full30k c0 runs (final-30k harness, same arm/seed) ==="
for d in /root/3dgs-publication-epic05/full30k/c0_s42_room /root/3dgs-publication-epic05/full30k/c0_s43_room /root/3dgs-publication-epic05/full30k/c0_s42_bicycle /root/3dgs-publication-epic05/full30k/c0_s43_bicycle; do
  if [ -f "$d/training_results.json" ]; then
    echo "--- $d"
    cat "$d/training_results.json"
  fi
done
echo "=== eval rows comparison (p12a room vs n1snap room) ==="
$PY - <<'EOF'
import json
def rows(p):
    d = json.load(open(p))
    return {r["step"]: r for r in d["eval_rows"]}
a = rows("/root/epic05_p12/runs/p12a/p12a_c0f_s42_room/results.json")
b = rows("/root/epic05_n1/runs/n1snap_c0f_s42_room/results.json")
print(f"{'step':>7} {'p12a_psnr':>10} {'n1_psnr':>9} {'d_psnr':>7} | {'p12a_n':>9} {'n1_n':>9} | {'p12a_l1':>8} {'n1_l1':>8}")
for s in sorted(set(a) & set(b)):
    if s >= 5000:
        print(f"{s:>7} {a[s]['psnr']:>10.3f} {b[s]['psnr']:>9.3f} {b[s]['psnr']-a[s]['psnr']:>7.3f} | "
              f"{a[s]['n_gaussians']:>9,} {b[s]['n_gaussians']:>9,} | {a[s]['l1']:>8.5f} {b[s]['l1']:>8.5f}")
# per-iter loss drift
da = json.load(open("/root/epic05_p12/runs/p12a/p12a_c0f_s42_room/results.json"))
db = json.load(open("/root/epic05_n1/runs/n1snap_c0f_s42_room/results.json"))
ia = {r["iter"]: r["loss"] for r in da["per_iter"]}
ib = {r["iter"]: r["loss"] for r in db["per_iter"]}
diffs = [abs(ia[k]-ib[k]) for k in sorted(set(ia)&set(ib))]
print(f"per-iter |dloss|: median={sorted(diffs)[len(diffs)//2]:.5f} max={max(diffs):.5f} n={len(diffs)}")
EOF
echo "=== bicycle progress ==="
tail -2 /root/epic05_n1/runs/n1snap_c0f_s42_bicycle/stdout.log
ls -la /root/epic05_n1/runs/n1snap_c0f_s42_room/*.pt

"""Find the first divergence between two training_curve.csv files
(columns: step, wall_time, psnr, ssim, lpips, l1, n_gaussians, tag)."""
import csv
import sys

a, b = sys.argv[1], sys.argv[2]
rows_a = [r for r in csv.DictReader(open(a)) if r["tag"] == "eval"]
rows_b = [r for r in csv.DictReader(open(b)) if r["tag"] == "eval"]
assert len(rows_a) == len(rows_b), f"length mismatch {len(rows_a)} vs {len(rows_b)}"
prev = None
for ra, rb in zip(rows_a, rows_b):
    assert ra["step"] == rb["step"]
    same_n = ra["n_gaussians"] == rb["n_gaussians"]
    if not same_n:
        print(f"FIRST N_DIVERGENCE at eval step {ra['step']}: "
              f"nA={ra['n_gaussians']} nB={rb['n_gaussians']} "
              f"(prev eval {prev})")
        break
    prev = ra["step"]
else:
    print("NO topology divergence across all recorded evals")
for ra, rb in zip(rows_a, rows_b):
    d = abs(float(ra["psnr"]) - float(rb["psnr"]))
    if d > 1e-9:
        print(f"first psnr diff at eval step {ra['step']}: "
              f"|dPSNR|={d:.3e} (A={ra['psnr']}, B={rb['psnr']})")
        break


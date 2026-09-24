"""R-13: drjohnson convergence-curve review (b1a vs c0, FINAL-30K seed 42).
Characterizes the +1.102 dB outlier: late divergence vs uniform shift vs eval artifact.
"""
import json

F30K = "/mnt/storage_pool/liaoyuanjun/final30k_runs"

rows = {}
for arm in ("b1a", "c0"):
    d = json.load(open(f"{F30K}/{arm}_drjohnson/results.json"))
    grid = {}
    for r in d.get("eval_rows", []):
        grid[r["step"]] = r["psnr"]
    rows[arm] = {"grid": grid, "final": d["final_eval"]["psnr"],
                 "ssim": d["final_eval"]["ssim"],
                 "lpips": d["final_eval"].get("lpips"),
                 "n": d["final_eval"]["n_gaussians"],
                 "wall": d["timing"]["total_wall_s"],
                 "n_eval_cams": d["final_eval"].get("n_eval_cameras") or d["final_eval"].get("n_cameras")}

g_b, g_c = rows["b1a"]["grid"], rows["c0"]["grid"]
steps = sorted(set(g_b) & set(g_c))
print(f"drjohnson eval trajectory (b1a vs c0, seed 42):")
print(f"{'step':>7s} {'b1a':>8s} {'c0':>8s} {'delta':>8s}")
for s in steps:
    print(f"{s:7d} {g_b[s]:8.3f} {g_c[s]:8.3f} {g_c[s]-g_b[s]:+8.3f}")
print(f"\nfinal_eval: b1a {rows['b1a']['final']:.3f} vs c0 {rows['c0']['final']:.3f} "
      f"(delta {rows['c0']['final']-rows['b1a']['final']:+.3f})")
print(f"N: b1a {rows['b1a']['n']:,} vs c0 {rows['c0']['n']:,} "
      f"(ratio {rows['c0']['n']/rows['b1a']['n']:.4f})")
print(f"eval cams: b1a={rows['b1a']['n_eval_cams']} c0={rows['c0']['n_eval_cams']}")

# characterize
deltas = [g_c[s] - g_b[s] for s in steps]
early = [d for s, d in zip(steps, deltas) if s <= 10000]
late = [d for s, d in zip(steps, deltas) if s >= 20000]
peak_i = max(range(len(steps)), key=lambda i: deltas[i])
peak_step = steps[peak_i]
# recovery jump = b1a's largest step-to-step PSNR gain after the peak
post = [(steps[i], g_b[steps[i]]) for i in range(len(steps)) if steps[i] >= peak_step]
jump = max((post[i + 1][1] - post[i][1], post[i][0], post[i + 1][0])
           for i in range(len(post) - 1)) if len(post) > 1 else (0.0, peak_step, peak_step)
print(f"\nmean delta early (<=10000): {sum(early)/len(early):+.3f}")
print(f"mean delta late  (>=20000): {sum(late)/len(late):+.3f}")
print(f"peak delta: {deltas[peak_i]:+.3f} @ step {peak_step}")
print(f"b1a recovery jump: {jump[0]:+.3f} ({jump[1]}->{jump[2]})")
print(f"final delta: {rows['c0']['final']-rows['b1a']['final']:+.3f}")

# P6 seed-persistence verdict (p6_results.json is authoritative once P6 lands)
try:
    p6 = json.load(open("/mnt/storage_pool/liaoyuanjun/pub_runs/p6_results.json"))
    dd = p6["drjohnson_deltas"]
    deltas_by_seed = dd if isinstance(dd, dict) else dict(zip(("42", "43", "44"), dd))
    seed_verdict = {
        "status": "COMPLETE",
        "deltas_by_seed": deltas_by_seed,
        "persistence": p6["drjohnson_persistent"],
    }
except (FileNotFoundError, KeyError):
    seed_verdict = {"status": "PENDING (c0s43/s43/s44/s44_drjohnson queued in P6)"}

verdict = {
    "claim": "R-13 drjohnson outlier characterization",
    "final_delta_db": rows["c0"]["final"] - rows["b1a"]["final"],
    "early_mean_delta": sum(early) / len(early),
    "late_mean_delta": sum(late) / len(late),
    "peak_delta_db": deltas[peak_i],
    "peak_step": peak_step,
    "b1a_recovery_jump_db": jump[0],
    "b1a_recovery_window": [jump[1], jump[2]],
    "n_ratio": rows["c0"]["n"] / rows["b1a"]["n"],
    "same_eval_cams": rows["b1a"]["n_eval_cams"] == rows["c0"]["n_eval_cams"],
    "seed_persistence_test": seed_verdict,
}
json.dump(verdict, open("/mnt/storage_pool/liaoyuanjun/pub_runs/r13_drjohnson_review.json", "w"), indent=1)
print("WROTE r13_drjohnson_review.json")

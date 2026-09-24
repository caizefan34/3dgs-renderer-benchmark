#!/usr/bin/env python3
"""E3: traceability - every headline citation in the reports matches a frozen JSON."""
import json
import sys

A = "C:/Users/36570/3dgs-renderer-benchmark/artifacts/publication/aggregates"
fails = []

def close(a, b, tol):
    return abs(a - b) <= tol

# 1. FINAL-30K headline
h = json.load(open(f"{A}/headline_c0_vs_b1a_f30k.json"))["aggregate"]
ok = close(h["speedup_geomean"], 1.0685, 0.00005) and close(h["mean_d_psnr"], 0.070, 0.0005)
print(f"[{'PASS' if ok else 'FAIL'}] headline 1.0685/+0.070 <- headline_c0_vs_b1a_f30k.json "
      f"({h['speedup_geomean']:.4f}/{h['mean_d_psnr']:+.4f})")
if not ok: fails.append("headline")

# 2. P2 ladder (A0 -> C0)
p = json.load(open(f"{A}/p2_c0_vs_a0.json"))["aggregate"]
ok = close(p["speedup_geomean"], 1.0045, 0.00005) and p["n_faster"] == 10 and close(p["mean_d_psnr"], 0.079, 0.0005)
print(f"[{'PASS' if ok else 'FAIL'}] P2 ladder 1.0045/10/13/+0.079 <- p2_c0_vs_a0.json "
      f"({p['speedup_geomean']:.4f}/{p['n_faster']}/13/{p['mean_d_psnr']:+.4f})")
if not ok: fails.append("p2")

# 3. B1 vs B1A (accutile)
p = json.load(open(f"{A}/p1_b1_vs_b1a.json"))["aggregate"]
ok = close(p["speedup_geomean"], 0.9662, 0.00005)
print(f"[{'PASS' if ok else 'FAIL'}] B1 geomean 0.9662 <- p1_b1_vs_b1a.json ({p['speedup_geomean']:.4f})")
if not ok: fails.append("b1")

# 4. B0 vs B1A
p = json.load(open(f"{A}/p1_b0_vs_b1a.json"))["aggregate"]
ok = close(p["speedup_geomean"], 0.4142, 0.00005)
print(f"[{'PASS' if ok else 'FAIL'}] B0 geomean 0.4142 <- p1_b0_vs_b1a.json ({p['speedup_geomean']:.4f})")
if not ok: fails.append("b0")

# 5. P6 per-seed geomeans
p6 = json.load(open(f"{A}/p6_results.json"))
geos = p6["per_seed_geomeans"]
ok = (close(geos["42"], 1.0705, 0.00005) and close(geos["43"], 1.0735, 0.00005)
      and close(geos["44"], 1.0561, 0.00005) and p6["drjohnson_persistent"] is True)
print(f"[{'PASS' if ok else 'FAIL'}] P6 geomeans 1.0705/1.0735/1.0561 + drjohnson REPRODUCIBLE "
      f"<- p6_results.json ({geos['42']:.4f}/{geos['43']:.4f}/{geos['44']:.4f})")
if not ok: fails.append("p6")

# 6. garden 3-seed
g = json.load(open(f"{A}/garden_seed3_table.json"))
deltas = [g["per_seed"][k]["effect_db"] for k in ("42", "43", "44")]
ok = (all(close(d, e, 0.0005) for d, e in zip(deltas, (0.642, 0.806, 1.132)))
      and g["sign_consistent"] is True)
print(f"[{'PASS' if ok else 'FAIL'}] garden 3-seed +0.642/+0.806/+1.132 sign-consistent "
      f"<- garden_seed3_table.json ({deltas[0]:+.3f}/{deltas[1]:+.3f}/{deltas[2]:+.3f})")
if not ok: fails.append("garden")

# 7. R-13 drjohnson review (seed-42 trajectory + completed seed test)
r = json.load(open(f"{A}/r13_drjohnson_review.json"))
ok = (close(r["peak_delta_db"], 3.37, 0.005) and r["peak_step"] == 15000
      and close(r["b1a_recovery_jump_db"], 4.74, 0.005)
      and r["seed_persistence_test"]["status"] == "COMPLETE"
      and r["seed_persistence_test"]["persistence"] is True
      and close(r["seed_persistence_test"]["deltas_by_seed"]["43"], 0.787, 0.0005))
print(f"[{'PASS' if ok else 'FAIL'}] R-13 peak +3.37@15K, recovery +4.74, seed test COMPLETE "
      f"<- r13_drjohnson_review.json ({r['peak_delta_db']:+.3f}@{r['peak_step']}, "
      f"s43 delta {r['seed_persistence_test']['deltas_by_seed']['43']:+.3f})")
if not ok: fails.append("r13")

print()
print("RESULT: " + ("ALL E3 TRACEABILITY CHECKS PASS" if not fails else f"FAILURES: {fails}"))
sys.exit(0 if not fails else 1)

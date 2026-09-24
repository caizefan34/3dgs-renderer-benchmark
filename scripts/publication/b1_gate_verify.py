"""B1_REPRODUCTION_PASS gate: compare pub_runs b1 gate runs (clean gsplat v1.5.3,
absgrad native, eps2d=0.1) against FINAL-30K b1a runs (same protocol + accutile)
on room/bicycle/garden.

Criteria (publication plan P1 gate):
  C1  run completed 30K with full outputs (results.json, final_status, timing,
      quality, training_results, training_curve.csv)
  C2  renderer metadata correct: base_commit 937e2991 pristine, accutile=False,
      absgrad=True, eps2d=0.1, binary sha 860936f2...
  C3  quality reproduction: |dPSNR(b1 - b1a_F30K)| <= 0.50 dB per scene
  C4  wall within +-10% of b1a_F30K
  C5  N_final within +-20% of b1a_F30K
  C6  timing_grade PUBLICATION and not contaminated
"""
import json, os

PUB = "/mnt/storage_pool/liaoyuanjun/pub_runs"
F30K = "/mnt/storage_pool/liaoyuanjun/final30k_runs"
SCENES = ["room", "bicycle", "garden"]

EXPECTED_SO = "860936f2c03e194b1fd8c13ef010dcf3cbf984c6d245281c53495159059b7c9b"

verdict_per_scene = {}
print(f"{'scene':9s} {'wall b1':>8s} {'wall b1a':>9s} {'ratio':>6s} {'PSNR b1':>8s} {'PSNR b1a':>8s} "
      f"{'dPSNR':>7s} {'N b1':>9s} {'N b1a':>9s} {'Nratio':>7s}")
all_ok = True
for s in SCENES:
    pb = os.path.join(PUB, f"b1_{s}", "results.json")
    fb = os.path.join(F30K, f"b1a_{s}", "results.json")
    checks = {}
    if not os.path.exists(pb):
        print(f"{s:9s} PENDING (no results.json yet)")
        all_ok = False
        verdict_per_scene[s] = {"status": "PENDING"}
        continue
    d1 = json.load(open(pb))
    d0 = json.load(open(fb))
    fe1, fe0 = d1["final_eval"], d0["final_eval"]
    w1, w0 = d1["timing"]["total_wall_s"], d0["timing"]["total_wall_s"]
    p1, p0 = fe1["psnr"], fe0["psnr"]
    n1, n0 = fe1["n_gaussians"], fe0["n_gaussians"]

    # C1 outputs
    need = ["results.json", "final_status.json", "timing.json", "quality.json",
            "training_results.json", "training_curve.csv"]
    checks["C1_outputs"] = all(os.path.exists(os.path.join(PUB, f"b1_{s}", f)) for f in need)
    # C2 metadata
    r = d1["renderer"]
    bi = d1["binary_identity"]
    checks["C2_metadata"] = (
        "937e2991" in str(r.get("base_commit", "")) and r.get("accutile") is False
        and r.get("absgrad") is True and abs((r.get("eps2d") or 0) - 0.1) < 1e-9
        and str(bi.get("so_sha256", "")).startswith(EXPECTED_SO[:16]))
    # C3/C4/C5
    checks["C3_dpsnr"] = abs(p1 - p0)
    checks["C3_pass"] = abs(p1 - p0) <= 0.50
    checks["C4_wall_ratio"] = w1 / w0
    checks["C4_pass"] = 0.90 <= w1 / w0 <= 1.10
    checks["C5_n_ratio"] = n1 / n0
    checks["C5_pass"] = 0.80 <= n1 / n0 <= 1.20
    # C6 grade
    checks["C6_grade"] = d1.get("timing_grade") == "PUBLICATION"
    st = json.load(open("/mnt/storage_pool/liaoyuanjun/pub_runs/pub_scheduler_state.json"))
    checks["C6_clean"] = not st.get("contaminated", {}).get(f"b1_{s}", False)

    ok = all(v for k, v in checks.items() if k.endswith("_pass")) and checks["C1_outputs"] \
        and checks["C2_metadata"] and checks["C6_grade"] and checks["C6_clean"]
    all_ok = all_ok and ok
    verdict_per_scene[s] = {"status": "PASS" if ok else "FAIL", "checks": checks}
    print(f"{s:9s} {w1:8.0f} {w0:9.0f} {w1/w0:6.3f} {p1:8.3f} {p0:8.3f} {p1-p0:+7.3f} "
          f"{n1:9d} {n0:9d} {n1/n0:7.3f}  {'PASS' if ok else 'FAIL'}")

final = ("B1_REPRODUCTION_PASS" if all_ok and
         all(v.get("status") == "PASS" for v in verdict_per_scene.values())
         else ("PENDING" if any(v.get("status") == "PENDING" for v in verdict_per_scene.values())
               else "B1_REPRODUCTION_FAIL"))
out = {"verdict": final, "per_scene": verdict_per_scene,
       "criteria": {"C3": "|dPSNR| <= 0.50 dB vs b1a FINAL-30K",
                    "C4": "wall within +-10% of b1a FINAL-30K",
                    "C5": "N_final within +-20% of b1a FINAL-30K",
                    "C6": "timing_grade PUBLICATION, uncontaminated"}}
with open("/mnt/storage_pool/liaoyuanjun/pub_runs/b1_gate_verdict.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nVERDICT: {final}")
print("WROTE /mnt/storage_pool/liaoyuanjun/pub_runs/b1_gate_verdict.json")

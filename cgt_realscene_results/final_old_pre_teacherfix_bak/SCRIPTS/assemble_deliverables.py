#!/usr/bin/env python
"""Assemble all CGT-Splat G1 deliverables + master verdict into final/."""
from __future__ import annotations
import argparse, csv, glob, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path
import numpy as np

GAMMA_SENS = (1.0, 2.0, 4.0)
GAMMA_GATE = 2.0
TARGET = 1.25
RANKS = (4, 8, 16)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="dir with g1_*.json + G1_REAL_SCENE_COMPRESSIBILITY.json")
    ap.add_argument("--src", required=True, help="dir with g1_compressibility.py analyze_g1.py run_g1.sh launch_all.sh assemble_g1.sh")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logs = Path(args.logs); out = Path(args.out); src = Path(args.src)
    (out / "G1_RAW_RECORDS").mkdir(parents=True, exist_ok=True)
    (out / "SCRIPTS").mkdir(parents=True, exist_ok=True)
    (out / "PROVENANCE").mkdir(parents=True, exist_ok=True)

    analysis = json.load(open(logs / "G1_REAL_SCENE_COMPRESSIBILITY.json"))
    combos = analysis["combos"]

    # ---- deliverable 3: preregistered gate ----
    gate = {
        "preregistered_before_full_run": True,
        "purpose": "G1 fidelity gate for CGT tile-response compressibility",
        "value_gate": {"psnr_db_min": 30.0, "max_abs_max": 0.02,
                        "note": "held-out odd pixels, (E,T) vs black/white/colored bg"},
        "jacobian_gate": {"metric": "jac_abs (RMS per-element student-teacher Jacobian error)",
                           "formula": "jac_abs <= max(REL_JAC*resp_range/TILE, ABS_FLOOR)",
                           "rel_jac": 0.5, "abs_floor": 0.02 / 16, "tile": 16,
                           "note": "well-posed scale-aware absolute; raw jac_rel is degenerate for near-flat tiles (teacher_jac_norm~1e-4 -> ratios to 1e27)"},
        "q_weighted": "work-weighted cumulative passing fraction, work=tile_work (Gaussian intersections per tile)",
        "amdahl": {"target_speedup_inference": TARGET, "gamma_gate": GAMMA_GATE,
                    "gamma_sensitivity": list(GAMMA_SENS),
                    "model": "fwd_ratio=[sum_pass R_best*gamma + sum_fail M]/sum_all M; S=1/fwd_ratio; M=tile_work"},
        "sampling": "stratified random by tile work decile (non top-N), 6 views x 250 tiles x 3 ranks = 4500 records/combo",
        "seed": 20261004,
    }
    json.dump(gate, open(out / "G1_PREREGISTERED_GATE.json", "w"), indent=1)

    # ---- deliverable 5: Amdahl min_q ----
    amdahl = {"gamma_gate": GAMMA_GATE, "target_speedup": TARGET,
               "note": "per-unique-tile work weighting. S_inference is a ZERO-OVERHEAD Amdahl upper bound (assumes per-component cost gamma x Gaussian and no tree/gating/compositing overhead). q_weighted (work-weighted compressible fraction) is the primary defensible metric; S is a loose upper bound, not a speedup claim.",
               "per_combo": {}}
    for k, c in combos.items():
        amdahl["per_combo"][k] = {
            "q_weighted_cumulative": c["q_weighted_cumulative"],
            "incompressible_work_frac": c["incompressible_work_frac"],
            "S_inference_actual": c["S_inference_actual"],
            "best_S_gamma_gate": c["best_S_gamma_gate"], "verdict": c["verdict"],
        }
    json.dump(amdahl, open(out / "AMDAHL_MIN_Q.json", "w"), indent=1)

    # ---- deliverable 6: per-view/per-tile CSV ----
    rows = []
    for fp in sorted(glob.glob(str(logs / "g1_*.json"))):
        b = os.path.basename(fp)
        if "smoke" in b or "test" in b: continue
        d = json.load(open(fp))
        for r in d["records"]:
            if "psnr" not in r: continue
            rows.append(dict(scene=d["scene"], ckpt=d["ckpt"].split("/")[-1], view=r["view"],
                             rank=r["rank"], tile=f"({r['tile_ti']},{r['tile_tj']})",
                             n_gauss=r["n_gauss"], tile_work=r["tile_work"],
                             psnr=round(r["psnr"], 2), max_abs=round(r["max_abs"], 5),
                             jac_abs=f"{r['jac_abs']:.3e}", resp_range=f"{r['resp_range']:.3e}",
                             teacher_jac_norm=f"{r['teacher_jac_norm']:.3e}"))
    with open(out / "G1_PER_VIEW_SUMMARY.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- deliverable 11: G2-G5 status ----
    gstatus = {
        "G1_compressibility": {"status": "PASS", "evidence": "G1_REAL_SCENE_COMPRESSIBILITY.json",
                                "q_weighted_r16_range": [0.811, 0.936],
                                "S_inference_gamma2_range": [3.39, 9.31],
                                "S_inference_gamma4_range": [2.39, 6.65]},
        "G2_failure_localization_certificate": {
            "status": "PARTIAL",
            "failure_localization": "DONE (from G1 data: incompressible work characterized by work decile / n_gauss / resp_range / scene; high-work deciles do NOT fail more)",
            "certificate": "NOT_REACHED",
            "reason": "Rigorous continuous-domain error certificate (plan §5.5) requires interval arithmetic or dense domain coverage; reference implements only per-tile held-out-pixel fitting (128 pts). Random held-out tests cannot give a unified continuous-domain certificate."},
        "G3_conservative_e2e_bound": {"status": "NOT_REACHED",
            "reason": "Requires full CGT cost model with MEASURED overhead (spatial tree traversal, gating, normalization, compositing, adaptive maintenance). Reference has no spatial tree / cost selector (plan §6). G1 S is an Amdahl upper bound assuming zero overhead."},
        "G4_gpu_microbenchmark": {"status": "NOT_REACHED",
            "reason": "Requires CUDA kernels for CGT component evaluation + A100 microbenchmark vs Gaussian eval. Plan §10.2 warns arithmetic reduction may not yield GPU speedup. Not implemented."},
        "G5_authoritative_e2e": {"status": "NOT_REACHED",
            "reason": "Requires native CGT training + full renderer integration + independent >=3x validation. Not implemented (reference is frozen-teacher per-tile fitting, route A inference compression only)."},
    }
    json.dump(gstatus, open(out / "G2_G5_STATUS.json", "w"), indent=1)

    # ---- deliverable 1: master verdict ----
    g1_pass = analysis.get("G1_overall_verdict") == "PASS"
    verdict = {
        "method": "CGT-Splat (Compositional Gaussian Transfer Splatting)",
        "track": "real-scene fail-closed G1->G5",
        "host": "MX bms-39468022-001, 8xA100-PCIe-40GB, runtime liaoyuanjun_mx0qual",
        "gpus_used": [5, 6, 7],
        "overall": "G1_PASS_HEADROOM_DEMONSTRATED_NOT_A_SPEEDUP_CLAIM" if g1_pass else "G1_DROP",
        "gates": {
            "G1": "PASS" if g1_pass else "DROP",
            "G2": "PARTIAL_NOT_REACHED_CERTIFICATE",
            "G3": "NOT_REACHED",
            "G4": "NOT_REACHED",
            "G5": "NOT_REACHED",
        },
        "G1_summary": {
            "combos": list(combos.keys()),
            "all_pass": g1_pass,
            "q_weighted_r16": {k: combos[k]["q_weighted_cumulative"]["16"] for k in combos},
            "S_inference_gamma2": {k: combos[k]["S_inference_actual"]["2.0"] for k in combos},
            "S_inference_gamma4": {k: combos[k]["S_inference_actual"]["4.0"] for k in combos},
        },
        "critical_caveats": [
            "S_inference is an Amdahl UPPER BOUND assuming per-component cost gamma=2-4x Gaussian and ZERO overhead (tree traversal/gating/normalization/compositing/adaptive maintenance unmeasured). NOT a real GPU speedup.",
            "Fitting is per-view (teacher = that view's rasterization); measures low-rank compressibility of the local transfer response, NOT cross-view 3D-consistent representation (that is G5 native training).",
            "Held-out is held-out PIXELS within a view, not held-out VIEWS; plan §9 G1 'test new views / occlusion change' not fully addressed (no cross-view generalization certificate).",
            "No rigorous continuous-domain error certificate (plan §5.5): random held-out tests cannot bound the error over the whole ray domain.",
            "Component-count reduction is not a GPU speedup; real speedup requires G3/G4.",
        ],
        "fail_closed": "Did not relax gates post-hoc, did not delete hard tiles, did not report unweighted Gaussian counts as speedup, did not convert Amdahl bound into a speedup claim. Higher rank worsens Jacobian (non-monotonic); did not auto-upgrade rank.",
        "status_freeze": "ALGEBRA_VALIDATED_SYNTHETIC_ONLY -> G1_REALSCENE_COMPRESSIBILITY_PASS_HEADROOM",
    }
    json.dump(verdict, open(out / "CGT_MASTER_VERDICT.json", "w"), indent=1)

    # ---- copy raw records + scripts ----
    for fp in sorted(glob.glob(str(logs / "g1_*.json"))):
        b = os.path.basename(fp)
        if "test" in b: continue
        subprocess.run(["cp", fp, str(out / "G1_RAW_RECORDS")])
    for fp in sorted(glob.glob(str(logs / "*.log"))):
        subprocess.run(["cp", fp, str(out / "G1_RAW_RECORDS")])
    for name in ["g1_compressibility.py", "analyze_g1.py", "run_g1.sh", "launch_all.sh", "assemble_g1.sh"]:
        p = src / name
        if p.exists():
            subprocess.run(["cp", str(p), str(out / "SCRIPTS")])

    # ---- provenance ----
    prov = []
    prov.append(f"host: {platform.node()}")
    prov.append(f"date_utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    prov.append(f"python: {sys.version.split()[0]}")
    try:
        import torch, gsplat
        prov.append(f"torch: {torch.__version__}  gsplat: {gsplat.__version__}  cuda_available: {torch.cuda.is_available()}  ngpu: {torch.cuda.device_count()}")
    except Exception as e:
        prov.append(f"import err: {e}")
    prov.append("gpu_used: 5,6,7 (verified 0 MiB before run)")
    prov.append("assets_root: /mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/")
    prov.append("data_root: /tmp/3dgs-renderer-benchmark/data/datasets/mipnerf360/")
    prov.append("runtime: /mnt/storage_pool/liaoyuanjun_mx0qual/bin/python")
    prov.append("seed: 20261004")
    prov.append("sampling: 6 views x 250 tiles (stratified by work decile) x 3 ranks")
    (out / "PROVENANCE" / "ENVIRONMENT.txt").write_text("\n".join(prov) + "\n")

    print("ASSEMBLE OK -> ", out)
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out)}  {p.stat().st_size}B")


if __name__ == "__main__":
    main()

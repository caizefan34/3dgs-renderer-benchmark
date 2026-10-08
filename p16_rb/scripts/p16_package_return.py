#!/usr/bin/env python3
"""P16-RB step 12/13 packaging: derived artifacts from the canonical records.

Inputs (all under --root):
  artifacts/P16_BASELINE_TABLE.json          (p16_aggregate_table.sh output)
  runs/<M>/<S>/P16_RUN.json                  canonical normalized records (15)
  curves/<M>/<S>.json                        TTQ curve replays (12 external)
  profiles/<M>/<S>_{mid,final}.json          bottleneck profiles (p16_profile_analyze.py)

Outputs (into artifacts/):
  P16_COMMON_METRICS.json     common metric records + manifest digests
  P16_BOTTLENECK_CENSUS.json  profiles + prereg section-15 classification + TTQ
  P16_VERDICT.json            baseline table + census classes + TTQ + notes
  P16_CONTAMINATION.json      per-run GPU/process capture audit
  MANIFEST_P16.json           sha256 of every required output file

TTQ convention: box(s) = C0 final metrics - section-10 tolerances; TTQ_method = first curve
point inside the box (training_elapsed_s), null if never; TTQ_C0 = C0 training_wall_s
(C0 reaches its own box at its final step; no C0 curve exists in this phase). Stdlib only.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

METHODS = ["FasterGS", "FasterGSFused", "FastGS", "SADGS"]
SCENES = ["room", "bicycle", "garden"]
PHASES = ["data", "projection_preprocess", "intersection_sort", "raster_forward", "loss",
          "raster_backward", "optimizer", "backward_optimizer_fused", "density_control", "other"]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/epic05_p16rb")
    args = ap.parse_args()
    R = Path(args.root)
    A = R / "artifacts"

    table = load(A / "P16_BASELINE_TABLE.json")
    runs = {}
    for m in METHODS + ["P12_C0"]:
        for s in SCENES:
            p = R / "runs" / m / s / "P16_RUN.json"
            if not p.exists():
                raise SystemExit(f"missing run record {p}")
            runs[(m, s)] = load(p)
    ref = {s: runs[("P12_C0", s)] for s in SCENES}

    def in_box(s, met):
        rr = ref[s]["metrics_common"]
        return (met["psnr"] >= rr["psnr"] - 0.10 and met["ssim"] >= rr["ssim"] - 0.003
                and met["lpips"] <= rr["lpips"] + 0.005)

    # ---------------------------------------------------------------- common metrics
    records = []
    for m in METHODS + ["P12_C0"]:
        for s in SCENES:
            r = runs[(m, s)]
            man = r.get("metrics_manifest_sha256") or {}
            digest = hashlib.sha256(json.dumps(sorted(man.items())).encode()).hexdigest() if man else None
            records.append({
                "method": m, "scene": s,
                "metrics_common": r["metrics_common"],
                "n_metrics_expected": r.get("n_metrics_expected"),
                "n_metrics_evaluated": r.get("n_metrics_evaluated"),
                "metrics_manifest_images": len(man),
                "metrics_manifest_digest": digest,
                "resolution": r["resolution"],
                "clean": r.get("clean"),
                "direct_comparable_resolution": r.get("direct_comparable_resolution"),
                "quality_admissible_vs_C0": (None if m == "P12_C0" else in_box(s, r["metrics_common"])),
                "run_record": str(R / "runs" / m / s / "P16_RUN.json"),
            })
    (A / "P16_COMMON_METRICS.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": "runs/<method>/<scene>/P16_RUN.json (metrics_common + manifest digests)",
        "box_rule": "PSNR >= C0-0.10 AND SSIM >= C0-0.003 AND LPIPS <= C0+0.005",
        "records": records,
    }, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- curves / TTQ
    curves, ttq = {}, {}
    for m in METHODS:
        for s in SCENES:
            cp = R / "curves" / m / f"{s}.json"
            if not cp.exists():
                raise SystemExit(f"missing curve {cp} (rerun2 not finished?)")
            c = load(cp)
            curves[(m, s)] = c
            hit = next((pt for pt in c["points"] if in_box(s, pt)), None)
            ttq[(m, s)] = {
                "ttq_s": hit["training_elapsed_s"] if hit else None,
                "ttq_fraction": hit["fraction"] if hit else None,
                "S_TTQ_vs_C0": round(ref[s]["training_wall_s"] / hit["training_elapsed_s"], 4) if hit else None,
                "curve": str(cp),
            }

    # ---------------------------------------------------------------- census
    p12_iters = {s: ref[s]["native_iterations"] for s in SCENES}
    census_rows = []
    for m in METHODS:
        for s in SCENES:
            stages = {}
            for st in ("mid", "final"):
                pp = R / "profiles" / m / f"{s}_{st}.json"
                if not pp.exists():
                    raise SystemExit(f"missing profile {pp} (rerun2 not finished?)")
                stages[st] = load(pp)
            fin = stages["final"]["median_ms"]
            tot = fin.get("total") or 0.0
            exec_ms = sum(fin.get(p) or 0.0 for p in
                          ("raster_forward", "raster_backward", "optimizer", "backward_optimizer_fused"))
            dens_ms = fin.get("density_control") or 0.0
            n_i = runs[(m, s)]["native_iterations"]
            exec_share = round(exec_ms / tot, 4) if tot else None
            dens_share = round(dens_ms / tot, 4) if tot else None
            it_frac = round(n_i / p12_iters[s], 4) if p12_iters[s] else None
            ttq_s = ttq[(m, s)]["ttq_s"]
            ttq_wall_frac = round(ttq_s / ref[s]["training_wall_s"], 4) if ttq_s else None
            if it_frac is not None and it_frac <= 0.5:
                cls, basis = "CONVERGENCE_DOMINATED", "iterations<=0.5*C0"
            elif ttq_wall_frac is not None and ttq_wall_frac <= 0.5:
                cls, basis = "CONVERGENCE_DOMINATED", "ttq<=0.5*C0_wall"
            elif exec_share is not None and exec_share >= 0.70 and it_frac >= 0.70:
                cls, basis = "EXECUTION_DOMINATED", "exec_share>=0.70 and iterations>=0.70*C0"
            else:
                cls, basis = "MIXED", None
            census_rows.append({
                "method": m, "scene": s,
                "p12_iterations": p12_iters[s], "native_iterations": n_i,
                "iterations_fraction_of_C0": it_frac,
                "exec_share_final": exec_share, "density_share_final": dens_share,
                "ttq_wall_fraction_of_C0": ttq_wall_frac,
                "classification": cls,
                "classification_basis": basis,
                "ttq": ttq[(m, s)],
                "profiles": stages,
                "protocol": "p16-prof-v1 attribution; medians over window iterations; "
                            "profiling replays are not headline wall time",
            })
    (A / "P16_BOTTLENECK_CENSUS.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "classification_rule": "prereg section 15: CONVERGENCE_DOMINATED if iterations <= 0.5*C0 "
                               "OR TTQ reached in <= 0.5*C0 wall; EXECUTION_DOMINATED "
                               "if renderer+backward/optimizer >= 0.70 of profile and iterations "
                               ">= 0.70*C0; else MIXED",
        "rows": census_rows,
    }, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- verdict
    scene_pareto = {}
    for s in SCENES:
        rows = [r for r in table["table"]
                if r["scene"] == s and r["method"] != "P12_C0" and r["quality_admissible"]
                and r["direct_comparable_resolution"] and r["clean"] and (r["speedup_vs_C0"] or 0) > 1.0]
        rows.sort(key=lambda r: -(r["speedup_vs_C0"] or 0))
        scene_pareto[s] = [{"method": r["method"], "speedup_vs_C0": r["speedup_vs_C0"],
                            "psnr": r["psnr"]} for r in rows]
    (A / "P16_VERDICT.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "verdict": table["verdict"],
        "strong_methods": table["strong_methods"],
        "methods": table["methods"],
        "scene_pareto_vs_C0": scene_pareto,
        "ttq": {f"{m}/{s}": ttq[(m, s)] for m in METHODS for s in SCENES},
        "census_classes": {f"{r['method']}/{r['scene']}": r["classification"] for r in census_rows},
        "baseline_table": str(A / "P16_BASELINE_TABLE.json"),
        "notes": [
            "verdict/table produced by bundle/p16rb_aggregate.py over the 15 canonical "
            "P16_RUN.json (staging aggregate_input_20260928; sha256 in INPUTS.sha256)",
            "garden_flatdata_corrupt records are corruption evidence, excluded per "
            "P16_DATA_HEAL_garden.json verdict_note; healed records are the results",
            "TTQ_C0 := C0 training_wall_s (C0 box = its own final metrics; no C0 curve this phase)",
            "curves are separate replays (CURVE_REPLAY), never mixed with primary wall time",
        ],
    }, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- contamination
    contam = []
    for m in METHODS + ["P12_C0"]:
        for s in SCENES:
            r = runs[(m, s)]
            def gpu_lines(key):
                out = []
                for ln in (r.get(key) or {}).get("stdout", "").splitlines():
                    parts = [x.strip() for x in ln.split(",")]
                    if len(parts) >= 3 and parts[2] == "NVIDIA A100-SXM4-80GB":
                        out.append({"index": parts[0], "uuid": parts[1], "used_mb": parts[3] if len(parts) > 3 else None})
                return out
            before, after = gpu_lines("gpu_before"), gpu_lines("gpu_after")
            proc_b = [x for x in (r.get("compute_processes_before") or {}).get("stdout", "").splitlines() if x.strip()]
            proc_a = [x for x in (r.get("compute_processes_after") or {}).get("stdout", "").splitlines() if x.strip()]
            contam.append({
                "method": m, "scene": s,
                "clean": r.get("clean"),
                "start_utc": r.get("start_utc"), "end_utc": r.get("end_utc"),
                "training_wall_s": r.get("training_wall_s"),
                "peak_vram_mb": r.get("peak_vram_mb"),
                "vram_pids_seen": r.get("vram_pids_seen"),
                "gpu_before": [g for g in before if g["index"] in ("2", "3")],
                "gpu_after": [g for g in after if g["index"] in ("2", "3")],
                "foreign_compute_processes_before": proc_b,
                "foreign_compute_processes_after": proc_a,
                "run_record": str(R / "runs" / m / s / "P16_RUN.json"),
            })
    (A / "P16_CONTAMINATION.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "policy": "one physical A100 (GPU 2, bcca92b9) per compared run; eval/metrics on GPU 3 "
                  "(431ac351); contamination flags adjudicated at finalize; contaminated runs "
                  "retained verbatim and rerun unchanged",
        "note": "nvidia-smi compute-apps may list foreign-namespace PIDs invisible to ps on this "
                "box (observed artifact); ownership is judged by GPU UUID",
        "runs": contam,
    }, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- manifest
    manifest = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "root": str(R), "files": {}}
    req = [
        A / "P16_SOURCE_IDENTITY.json", A / "P16_CONFIG_PROVENANCE.json",
        A / "P16_ENVIRONMENTS.json", A / "P16_RESOLUTION_SPLIT_AUDIT.json",
        A / "P16_BASELINE_TABLE.json", A / "P16_COMMON_METRICS.json",
        A / "P16_BOTTLENECK_CENSUS.json", A / "P16_VERDICT.json",
        A / "P16_CONTAMINATION.json", A / "P16_DATA_HEAL_garden.json",
    ]
    for m in METHODS + ["P12_C0"]:
        for s in SCENES:
            req.append(R / "runs" / m / s / "P16_RUN.json")
    for m in METHODS:
        for s in SCENES:
            req.append(R / "curves" / m / f"{s}.json")
            req.append(R / "profiles" / m / f"{s}_mid.json")
            req.append(R / "profiles" / m / f"{s}_final.json")
    report = A / "P16_REPORT.md"
    if report.exists():
        req.append(report)
    for p in req:
        if not p.exists():
            raise SystemExit(f"manifest target missing: {p}")
        manifest["files"][str(p.relative_to(R)).replace("\\", "/")] = {
            "sha256": sha256_file(p), "bytes": p.stat().st_size}
    (A / "MANIFEST_P16.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"verdict": table["verdict"], "strong_methods": table["strong_methods"],
                      "manifest_files": len(manifest["files"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

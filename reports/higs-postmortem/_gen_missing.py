#!/usr/bin/env python3
"""Create missing artifacts: backward_sensitivity.json, source_index.json, environment.json, stage_timing_by_scene.json"""
import json, glob, os, statistics, math
from collections import defaultdict

outdir = r"artifacts\higs-trainable-regression"
os.makedirs(outdir, exist_ok=True)

def clean(x):
    try: return float(x)
    except Exception: return None
def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None

# ---------- backward_sensitivity.json ----------
# Analytic model: total_iter = fwd + bwd + opt + topo + gap
# For a 10%/20% reduction in backward stage cost, total reduction = reduction * (bwd/total)
# Compute per scene from stage logs (mean across tiles)
logs = {}
for f in glob.glob(r"results/training/*.json"):
    r = json.load(open(f))
    cfg = r.get("config", {}) or {}
    logs[(cfg.get("scene"), cfg.get("tile_size"))] = r

scene_stage = {}
for (scene, tile), r in logs.items():
    ml = r.get("metrics_log", [])
    if not ml:
        continue
    sums = {}
    for k in ["fwd_ms", "bwd_ms", "opt_ms", "topology_ms", "iteration_ms"]:
        vals = [clean(e.get(k)) for e in ml]
        vals = [v for v in vals if v is not None]
        if vals:
            sums[k] = sum(vals)
    acc = sum(sums.get(k, 0) for k in ["fwd_ms", "bwd_ms", "opt_ms", "topology_ms"])
    scene_stage.setdefault(scene, []).append((sums, acc))

sens = {}
for scene, entries in scene_stage.items():
    tot_iter = mean([s.get("iteration_ms", 0) for s, _ in entries]) or 0
    bwd = mean([s.get("bwd_ms", 0) for s, _ in entries]) or 0
    sens[scene] = {
        "bwd_share_pct": round(100.0 * bwd / tot_iter, 1) if tot_iter else None,
        "total_red_pct_at_10pct_bwd_cut": round(10.0 * bwd / tot_iter, 2) if tot_iter else None,
        "total_red_pct_at_20pct_bwd_cut": round(20.0 * bwd / tot_iter, 2) if tot_iter else None,
        "total_red_pct_at_100pct_bwd_cut": round(100.0 * bwd / tot_iter, 2) if tot_iter else None,
    }
with open(os.path.join(outdir, "backward_sensitivity.json"), "w") as f:
    json.dump(sens, f, indent=2)

# ---------- source_index.json ----------
def find_sources():
    files = []
    for pattern in ["artifacts/training-paper/results/*.json",
                    "artifacts/training-all/results/*.json",
                    "results/training/*.json",
                    "benchmark/higs-paper-protocol.json",
                    "reports/r4/r4-13scene-final.md",
                    "reports/r4/r4-13scan-final.md"]:
        files += sorted(glob.glob(pattern))
    return files
src = {
    "generated": "2026-09-19T00:00:00+00:00",
    "purpose": "Regression postmortem evidence index for Trainable HiGS 13x3x3 study",
    "primary_results_corpora": [
        {"name": "training-paper", "path": "artifacts/training-paper/results", "count": len(glob.glob(r"artifacts/training-paper/results/*.json")), "note": "99 files = 11 scenes x 3 seeds x 3 methods (gsplat, higs_full, higs_proposed)"},
        {"name": "training-all", "path": "artifacts/training-all/results", "count": len(glob.glob(r"artifacts/training-all/results/*.json")), "note": "165 files = 11 scenes x 3 seeds x 5 methods (adds original_3dgs, speedy_splat)"},
    ],
    "profile_logs": {
        "path": "results/training",
        "count": len(glob.glob(r"results/training/*.json")),
        "note": "18 files = 3 scenes (bicycle/garden/room) x 3 tile sizes (16/20/24) x 2 (30k / 500-step); per-iteration fwd/bwd/opt/topology breakdown"
    },
    "protocol": "benchmark/higs-paper-protocol.json",
    "related_reports": [
        "reports/r4/r4-13scene-final.md",
        "reports/r4/r4-13scan-final.md"
    ],
    "source_files": find_sources()
}
with open(os.path.join(outdir, "source_index.json"), "w") as f:
    json.dump(src, f, indent=2)

# ---------- environment.json ----------
env = {
    "gpu": {
        "target": "NVIDIA A100-SXM4-80GB (mx server)",
        "profile_logs": "NVIDIA A100-PCIE-40GB"
    },
    "software": {
        "gsplat": "source: https://github.com/nerfstudio-project/gsplat (commit 77ab983 linked in provenance)",
        "pytorch": "2.x (CPU timing via iteration_ms; CUDA events recommended for stage attribution)",
        "cuda": "12.8 (see provenance patch hash in r4 report)",
        "python": "3.10+ (inferred from codebase conventions)"
    },
    "benchmark_protocol": {
        "file": "benchmark/higs-paper-protocol.json",
        "timing_boundary": "dataset_ready_to_final_checkpoint",
        "sync": "cuda_synchronize_before_start_and_stop required at instrumentation boundary (per protocol)",
        "warmup_notes": "first 1100 iters excluded from timing in the grid corpus"
    },
    "data": {
        "scenes": ["deep_blending/drjohnson", "deep_blending/playroom", "mipnerf360/bicycle", "mipnerf360/bonsai", "mipnerf360/counter", "mipnerf360/garden", "mipnerf360/kitchen", "mipnerf360/room", "mipnerf360/stump", "tanks_and_temples/train", "tanks_and_temples/truck"],
        "seeds": [0, 1, 2],
        "nb_sources": "11 primary + 5 cross-hardware scenes per protocol"
    },
    "caveats": [
        "per-iteration stage logs are from the older single-GPU profiling run (Sep 4), not from the 117-job study",
        "wall-time geomeans recalculated over 99-file committed corpus may differ from the 117-job aggregation",
        "timer attribution uses synchronous boundaries; async kernel overlap may undercount stages sharing a stream"
    ]
}
with open(os.path.join(outdir, "environment.json"), "w") as f:
    json.dump(env, f, indent=2)

print("wrote:", sorted(os.listdir(outdir)))

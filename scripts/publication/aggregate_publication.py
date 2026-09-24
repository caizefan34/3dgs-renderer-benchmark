#!/usr/bin/env python3
"""PUBLICATION aggregation engine: scan all frozen run trees, build the master
run table, and emit per-phase aggregates. Idempotent; re-run as runs land.

Sources (frozen artifacts only -- never transcribed numbers):
  FINAL-30K: /mnt/storage_pool/liaoyuanjun/final30k_runs   (b1a_*, c0_*)
  PUB:       /mnt/storage_pool/liaoyuanjun/pub_runs         (all publication arms)

Output: artifacts/publication/aggregates/runs_master.json (+ print summary)

§19 discipline: speedup = geomean of per-scene RATIO; time reduction = 1-1/ratio;
per-scene reduction geomean is a SEPARATELY LABELED metric. §20: quality wording
"quality-neutral within the matched benchmark". Contaminated runs are recorded,
never dropped, and their wall is excluded from publication timing aggregates.
"""
import json
import math
import os

FINAL30K = "/mnt/storage_pool/liaoyuanjun/final30k_runs"
PUB = "/mnt/storage_pool/liaoyuanjun/pub_runs"
OUT_DIR = "/mnt/storage_pool/liaoyuanjun/pubphase/aggregates"

ALL13 = ["bicycle", "bonsai", "counter", "drjohnson", "flowers", "garden",
         "kitchen", "playroom", "room", "stump", "train", "treehill", "truck"]

# key prefix -> (variant, default_seed, default_eps2d, eps2d_overridable)
PREFIX_MAP = [
    ("b1ae03", "b1a", 42, 0.3, True),    # P4 corner (explicit 0.3)
    ("b1as43", "b1a", 43, 0.1, False),   # seed-43 b1a
    ("b1as44", "b1a", 44, 0.1, False),
    ("b1s43", "b1", 43, 0.1, False),
    ("b1s44", "b1", 44, 0.1, False),
    ("c0s43", "c0", 43, 0.3, False),
    ("c0s44", "c0", 44, 0.3, False),
    ("c0e01", "c0", 42, 0.1, True),      # P4 corner (explicit 0.1)
    ("b1a", "b1a", 42, 0.1, True),       # FINAL-30K (default 0.1)
    ("b1", "b1", 42, 0.1, True),
    ("b0", "b0", 42, None, False),
    ("a0", "a0", 42, 0.3, False),
    ("a1", "a1", 42, 0.3, False),
    ("a2", "a2", 42, 0.3, False),
    ("c0", "c0", 42, 0.3, True),         # FINAL-30K (default 0.3)
]


def classify(dirname):
    scene = None
    for s in ALL13:
        if dirname.endswith("_" + s):
            scene = s
            break
    if scene is None:
        return None
    prefix = dirname[: -(len(scene) + 1)]
    for p, variant, seed, eps2d, _ in PREFIX_MAP:
        if prefix == p:
            return {"prefix": prefix, "variant": variant, "scene": scene,
                    "seed": seed, "eps2d_default": eps2d}
    return None


def load_run(root, dirname, cls):
    rj = os.path.join(root, dirname, "results.json")
    if not os.path.exists(rj):
        return None
    try:
        d = json.load(open(rj))
    except Exception:
        return None
    fe = d["final_eval"]
    t = d["timing"]
    ph = t.get("phase_ms", {})
    eps2d_eff = d.get("renderer", {}).get("eps2d", cls["eps2d_default"])
    return {
        "run_id": dirname,
        "source": "final30k" if root == FINAL30K else "pub",
        "variant": cls["variant"],
        "scene": cls["scene"],
        "seed": d.get("seed", cls["seed"]),
        "eps2d": eps2d_eff,
        "eps2d_override": d.get("eps2d_override"),
        "arm": d.get("arm"),
        "wall_s": t.get("total_wall_s"),
        "mean_iter_ms": t.get("mean_iter_ms"),
        "psnr": fe["psnr"], "ssim": fe["ssim"], "lpips": fe.get("lpips"),
        "l1": fe.get("l1"),
        "n_gaussians": fe["n_gaussians"],
        "initial_n": d.get("initial_N"),
        "peak_vram_gb": d.get("peak_vram_gb"),
        "timing_grade": d.get("timing_grade"),
        "binary_sha": d.get("binary_identity", {}).get("so_sha256", "")[:16],
        "phase_ms": {k: (v.get("mean") if isinstance(v, dict) else v)
                     for k, v in ph.items()},
        "eval_grid": [(r["step"], r["psnr"]) for r in d.get("eval_rows", [])],
        "n_iters": t.get("n_iters"),
    }


def scan(root):
    out = []
    if not os.path.isdir(root):
        return out
    for dirname in sorted(os.listdir(root)):
        cls = classify(dirname)
        if cls is None:
            continue
        r = load_run(root, dirname, cls)
        if r and r.get("n_iters") == 30000 and r.get("psnr") is not None:
            out.append(r)
    return out


def mark_contaminated(runs):
    st_path = os.path.join(PUB, "pub_scheduler_state.json")
    if not os.path.exists(st_path):
        return
    st = json.load(open(st_path))
    contam = st.get("contaminated", {})
    for r in runs:
        if contam.get(r["run_id"], False):
            r["contaminated"] = True
        else:
            r.setdefault("contaminated", False)


def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def pair_table(runs, base_v, cand_v, seed=42, eps2d_b=None, eps2d_c=None,
               require_clean=True):
    """Per-scene paired comparison cand vs base (matched seed/eps2d where given)."""
    def find(v, s, e):
        for r in runs:
            if (r["variant"] == v and r["scene"] == s and r["seed"] == seed
                    and (e is None or r["eps2d"] == e)):
                return r
        return None
    rows = []
    for s in ALL13:
        b = find(base_v, s, eps2d_b)
        c = find(cand_v, s, eps2d_c)
        if not b or not c:
            rows.append({"scene": s, "status": "pending"})
            continue
        clean_b = (not b["contaminated"]) and b["timing_grade"] == "PUBLICATION"
        clean_c = (not c["contaminated"]) and c["timing_grade"] == "PUBLICATION"
        row = {
            "scene": s, "status": "ok",
            "wall_base": b["wall_s"], "wall_cand": c["wall_s"],
            # S19 orientation: speedup > 1 means the CANDIDATE is faster
            # (wall_base / wall_cand, as in the FINAL-30K headline 1.0685x)
            "speedup": (b["wall_s"] / c["wall_s"]) if (clean_b and clean_c) else None,
            "psnr_base": b["psnr"], "psnr_cand": c["psnr"],
            "d_psnr": c["psnr"] - b["psnr"],
            "d_ssim": c["ssim"] - b["ssim"],
            "d_lpips": (c["lpips"] - b["lpips"]) if (c.get("lpips") is not None and b.get("lpips") is not None) else None,
            "n_base": b["n_gaussians"], "n_cand": c["n_gaussians"],
            "n_ratio": c["n_gaussians"] / b["n_gaussians"],
            "timing_clean_pair": clean_b and clean_c,
            "base_id": b["run_id"], "cand_id": c["run_id"],
        }
        rows.append(row)
    ok = [r for r in rows if r["status"] == "ok"]
    speedups = [r["speedup"] for r in ok if r.get("speedup")]
    reductions = [1 - 1 / x for x in speedups]
    agg = {
        "n_pairs": len(ok),
        "speedup_geomean": geomean(speedups),
        # per-scene reduction geomean (S19 separately-labeled metric); only
        # defined when the candidate is faster everywhere (all reductions > 0)
        "reduction_geomean_pct": (geomean(reductions) * 100)
                                 if (speedups and all(x > 0 for x in reductions)) else None,
        "mean_d_psnr": sum(r["d_psnr"] for r in ok) / len(ok) if ok else None,
        "mean_d_ssim": sum(r["d_ssim"] for r in ok) / len(ok) if ok else None,
        "mean_d_lpips": sum(r["d_lpips"] for r in ok if r.get("d_lpips") is not None) / max(1, len([r for r in ok if r.get("d_lpips") is not None])),
        "n_faster": sum(1 for r in ok if r.get("speedup") and r["speedup"] > 1.0),
        "n_slower": sum(1 for r in ok if r.get("speedup") and r["speedup"] < 1.0),
        "n_geo_ratio": geomean([r["n_ratio"] for r in ok]),
    }
    return {"rows": rows, "aggregate": agg,
            "note": "speedup=geomean of per-scene ratios; reduction_geomean_pct is the "
                    "separately-labeled per-scene reduction aggregate (S19); contaminated "
                    "pairs excluded from timing (recorded, not dropped)"}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    runs = scan(FINAL30K) + scan(PUB)
    mark_contaminated(runs)

    master = {
        "generated": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "n_runs": len(runs),
        "protocol": "30K, seed per run, max_side 1920, COLMAP, 0.8*L1+0.2*(1-SepSSIM)",
        "runs": runs,
    }
    with open(os.path.join(OUT_DIR, "runs_master.json"), "w") as f:
        json.dump(master, f, indent=1)

    # headline P1/P2 pair tables (matched seed 42)
    tables = {
        "p1_b1_vs_b1a": pair_table(runs, "b1a", "b1", eps2d_b=0.1, eps2d_c=0.1),
        "p1_b0_vs_b1a": pair_table(runs, "b1a", "b0", eps2d_b=0.1, eps2d_c=None),
        "p1_c0_vs_b1": pair_table(runs, "b1", "c0", eps2d_b=0.1, eps2d_c=0.3),
        "p2_a1_vs_a0": pair_table(runs, "a0", "a1", eps2d_b=0.3, eps2d_c=0.3),
        "p2_a2_vs_a1": pair_table(runs, "a1", "a2", eps2d_b=0.3, eps2d_c=0.3),
        "p2_c0_vs_a0": pair_table(runs, "a0", "c0", eps2d_b=0.3, eps2d_c=0.3),
        "p2_c0_vs_a2": pair_table(runs, "a2", "c0", eps2d_b=0.3, eps2d_c=0.3),
        "headline_c0_vs_b1a_f30k": pair_table(runs, "b1a", "c0", eps2d_b=0.1, eps2d_c=0.3),
    }
    for name, t in tables.items():
        with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as f:
            json.dump(t, f, indent=1)

    print(f"master: {len(runs)} runs -> {OUT_DIR}/runs_master.json")
    by_variant = {}
    for r in runs:
        by_variant.setdefault((r["variant"], r["seed"], r["eps2d"]), []).append(r["scene"])
    for (v, s, e), scenes in sorted(by_variant.items(), key=lambda x: str(x[0])):
        print(f"  {v:5s} seed={s} eps2d={e}: {len(scenes):2d} scenes")
    print("\nheadline check (C0 vs B1A, FINAL-30K reuse):")
    h = tables["headline_c0_vs_b1a_f30k"]["aggregate"]
    print(f"  pairs={h['n_pairs']} speedup_geomean={h['speedup_geomean']:.4f} "
          f"mean_d_psnr={h['mean_d_psnr']:+.3f}")


if __name__ == "__main__":
    main()

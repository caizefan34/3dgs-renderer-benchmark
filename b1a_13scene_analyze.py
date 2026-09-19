#!/usr/bin/env python3
"""Collect all 26 B1/B1A training_results.json from /dev/shm/accutile30k and
produce the machine-readable results.json plus data structures for reports."""
import json, os, sys, math
from pathlib import Path
from collections import defaultdict

OUT_BASE = "/dev/shm/accutile30k"
LOCAL_OUT = Path(__file__).resolve().parent / "reports" / "accutile30k"

MIPNERF360 = ["bicycle", "bonsai", "counter", "flowers", "garden",
              "kitchen", "room", "stump", "treehill"]
TANKSTEMPLES = ["train", "truck"]
DEEPBLENDING = ["drjohnson", "playroom"]

SCENE_DATASET = {}
for s in MIPNERF360: SCENE_DATASET[s] = "Mip-NeRF360"
for s in TANKSTEMPLES: SCENE_DATASET[s] = "Tanks & Temples"
for s in DEEPBLENDING: SCENE_DATASET[s] = "Deep Blending"

ALL_SCENES = MIPNERF360 + TANKSTEMPLES + DEEPBLENDING


def geomean(vals):
    vals = [v for v in vals if v is not None and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def arith_mean(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def load_run(scene, method):
    path = os.path.join(OUT_BASE, scene, method, "training_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def collect():
    results = {}
    for scene in ALL_SCENES:
        b1 = load_run(scene, "b1")
        b1a = load_run(scene, "b1a")
        results[scene] = {"b1": b1, "b1a": b1a}

    per_scene = []
    for scene in ALL_SCENES:
        b1 = results[scene]["b1"]
        b1a = results[scene]["b1a"]
        ds = SCENE_DATASET[scene]
        entry = {"scene": scene, "dataset": ds}

        if b1 and b1a:
            # Final eval (all cameras)
            b1_fe = b1.get("final_eval", {})
            b1a_fe = b1a.get("final_eval", {})
            entry["b1_psnr"] = b1_fe.get("psnr")
            entry["b1a_psnr"] = b1a_fe.get("psnr")
            entry["b1_ssim"] = b1_fe.get("ssim")
            entry["b1a_ssim"] = b1a_fe.get("ssim")
            entry["b1_lpips"] = b1_fe.get("lpips")
            entry["b1a_lpips"] = b1a_fe.get("lpips")
            entry["dpsnr"] = (entry["b1a_psnr"] - entry["b1_psnr"]) if entry["b1_psnr"] is not None and entry["b1a_psnr"] is not None else None
            entry["dssim"] = (entry["b1a_ssim"] - entry["b1_ssim"]) if entry["b1_ssim"] is not None and entry["b1a_ssim"] is not None else None
            entry["dlpips"] = (entry["b1a_lpips"] - entry["b1_lpips"]) if entry["b1_lpips"] is not None and entry["b1a_lpips"] is not None else None

            # Timing
            b1_t = b1.get("timing", {})
            b1a_t = b1a.get("timing", {})
            entry["b1_total_wall_s"] = b1_t.get("total_wall_s")
            entry["b1a_total_wall_s"] = b1a_t.get("total_wall_s")
            entry["b1_mean_iter_ms"] = b1_t.get("mean_iter_ms")
            entry["b1a_mean_iter_ms"] = b1a_t.get("mean_iter_ms")
            entry["b1_median_iter_ms"] = b1_t.get("median_iter_ms")
            entry["b1a_median_iter_ms"] = b1a_t.get("median_iter_ms")
            entry["b1_steady_iter_ms"] = b1_t.get("steady_state_mean_iter_ms")
            entry["b1a_steady_iter_ms"] = b1a_t.get("steady_state_mean_iter_ms")
            entry["training_speedup"] = (entry["b1_total_wall_s"] / entry["b1a_total_wall_s"]) if entry["b1_total_wall_s"] and entry["b1a_total_wall_s"] else None
            entry["iter_speedup"] = (entry["b1_steady_iter_ms"] / entry["b1a_steady_iter_ms"]) if entry["b1_steady_iter_ms"] and entry["b1a_steady_iter_ms"] else None

            # Gaussian counts
            entry["b1_final_N"] = b1.get("final_N")
            entry["b1a_final_N"] = b1a.get("final_N")
            entry["dN"] = (entry["b1a_final_N"] - entry["b1_final_N"]) if entry["b1_final_N"] is not None and entry["b1a_final_N"] is not None else None

            # Topology events
            entry["b1_clones"] = b1.get("total_clones")
            entry["b1a_clones"] = b1a.get("total_clones")
            entry["b1_splits"] = b1.get("total_splits")
            entry["b1a_splits"] = b1a.get("total_splits")
            entry["b1_prunes"] = b1.get("total_prunes")
            entry["b1a_prunes"] = b1a.get("total_prunes")

            # N_GS(t) trajectory at checkpoint iters
            b1_ck = b1.get("training_metrics", {}).get("checkpoints", {})
            b1a_ck = b1a.get("training_metrics", {}).get("checkpoints", {})
            entry["ngs_trajectory"] = {}
            for it in ["5000", "10000", "15000", "20000", "25000", "30000"]:
                b1_n = b1_ck.get(it, {}).get("n_gaussians")
                b1a_n = b1a_ck.get(it, {}).get("n_gaussians")
                entry["ngs_trajectory"][it] = {"b1": b1_n, "b1a": b1a_n,
                                                "dN": (b1a_n - b1_n) if b1_n and b1a_n else None}

            # PSNR/SSIM trajectory
            entry["psnr_trajectory"] = {}
            for it in ["5000", "10000", "15000", "20000", "25000", "30000"]:
                entry["psnr_trajectory"][it] = {
                    "b1": b1_ck.get(it, {}).get("psnr"),
                    "b1a": b1a_ck.get(it, {}).get("psnr"),
                }

            entry["status"] = "complete"
        elif b1 or b1a:
            entry["status"] = "partial"
        else:
            entry["status"] = "missing"
        per_scene.append(entry)

    # Dataset aggregation
    def agg(dataset_scenes, label):
        complete = [e for e in per_scene if e["dataset"] == label and e.get("status") == "complete"]
        speedups = [e["training_speedup"] for e in complete if e.get("training_speedup")]
        return {
            "n_scenes": len(complete),
            "geomean_training_speedup": geomean(speedups),
            "arith_mean_training_speedup": arith_mean(speedups),
            "mean_dpsnr": arith_mean([e["dpsnr"] for e in complete if e.get("dpsnr") is not None]),
            "mean_dssim": arith_mean([e["dssim"] for e in complete if e.get("dssim") is not None]),
            "mean_dlpips": arith_mean([e["dlpips"] for e in complete if e.get("dlpips") is not None]),
            "scenes_complete": len(complete),
        }

    mipnerf360_agg = agg(MIPNERF360, "Mip-NeRF360")
    tt_agg = agg(TANKSTEMPLES, "Tanks & Temples")
    db_agg = agg(DEEPBLENDING, "Deep Blending")

    all_complete = [e for e in per_scene if e.get("status") == "complete"]
    all_speedups = [e["training_speedup"] for e in all_complete if e.get("training_speedup")]
    all13_agg = {
        "n_scenes": len(all_complete),
        "geomean_training_speedup": geomean(all_speedups),
        "arith_mean_training_speedup": arith_mean(all_speedups),
        "mean_dpsnr": arith_mean([e["dpsnr"] for e in all_complete if e.get("dpsnr") is not None]),
        "mean_dssim": arith_mean([e["dssim"] for e in all_complete if e.get("dssim") is not None]),
        "mean_dlpips": arith_mean([e["dlpips"] for e in all_complete if e.get("dlpips") is not None]),
    }

    return {
        "per_scene": per_scene,
        "dataset_aggregation": {
            "Mip-NeRF360": mipnerf360_agg,
            "Tanks & Temples": tt_agg,
            "Deep Blending": db_agg,
            "all_13": all13_agg,
        },
        "raw": results,
    }


def main():
    data = collect()
    n_complete = sum(1 for e in data["per_scene"] if e.get("status") == "complete")
    print(f"Scenes complete: {n_complete}/13")
    for e in data["per_scene"]:
        sp = e.get("training_speedup")
        dpsnr = e.get("dpsnr")
        print(f"  {e['scene']:12s} {e['status']:8s} speedup={sp if sp is not None else 'N/A':>6} dpsnr={dpsnr if dpsnr is not None else 'N/A'}")

    out_path = LOCAL_OUT / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Strip 'raw' for the saved file to keep it manageable; keep per_scene + aggregation
    save = {"per_scene": data["per_scene"], "dataset_aggregation": data["dataset_aggregation"]}
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved {out_path}")
    # Also save full raw for provenance
    with open(LOCAL_OUT / "results_raw.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {LOCAL_OUT / 'results_raw.json'}")


if __name__ == "__main__":
    main()

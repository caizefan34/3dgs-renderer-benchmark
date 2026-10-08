"""SplatMG SG0 finalize: aggregate per-cell JSONs into the frozen top-level
outputs, run sg0_analyze.py (the analyzer owns the verdict), and write the
report + SHA256 manifest. No threshold tuning; the analyzer is authoritative.
"""
import os
import sys
import json
import time
import hashlib
import subprocess
import numpy as np

sys.path.insert(0, "/root/epic05_sg0")
import sg0_lib as G

ROOT = "/root/epic05_sg0"
CELLDIR = os.path.join(ROOT, "cells")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=G.L.json_default)


def main():
    t0 = time.time()
    def log(*a):
        print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

    cells = {}
    missing = []
    for scene, ckpt in G.CELLS:
        cell = f"{scene}@{ckpt}"
        p = os.path.join(CELLDIR, f"SG0_CELL_{cell}.json")
        if os.path.isfile(p):
            cells[cell] = json.load(open(p))
            log("loaded", cell)
        else:
            missing.append(cell)
            log("MISSING", cell)
    if missing:
        log("missing cells (halted/undone):", missing)

    # ---- SOURCE_IDENTITY (renderer + checkpoints) ----
    env = G.L.bootstrap_c0()
    ckpt_identity = {}
    for scene, ckpt in G.CELLS:
        cp = f"/root/epic05_n1/runs/n1snap_c0f_s42_{scene}/ckpt_{ckpt}.pt"
        ckpt_identity[f"{scene}@{ckpt}"] = {
            "ckpt_path": cp, "ckpt_sha256": sha256_file(cp)}
    source_identity = {
        "campaign": "MOONSHOT0-SG0", "track": "SG0",
        "study": "SplatMG future-displacement coarse-space oracle",
        "frozen_date_prereg": "2026-09-30",
        "baseline": "P12/C0 (room+bicycle, 20K+25K)",
        "renderer": {
            "arm": env["arm"],
            "so_sha256": env["so_sha256"],
            "so_path": env["so_path"],
            "core_sha256": env["core_sha256"],
            "core_path": env["core_path"],
            "tile_size": 16, "eps2d": 0.3, "packed": False, "absgrad": True,
            "sh_degree": 3,
        },
        "loss": "0.8*L1 + 0.2*(1-SSIM) fused-ssim padding=same (full resolution)",
        "checkpoints": ckpt_identity,
        "known_identities_from_prereg": {
            "C0_extension_d26cccf": "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11",
            "core_gsplat_cuda_so_935ea6c0": "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e",
            "room@25K_ckpt_b31e5a6b": "b31e5a6b0e733911b736bbad2e4f9d4d804d5929655f64efb51a9e5132fc047d",
        },
    }
    jdump(source_identity, os.path.join(ROOT, "SG0_SOURCE_IDENTITY.json"))

    # ---- ENVIRONMENT ----
    import torch
    environment = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "host": os.uname().nodename,
        "timestamp": time.time(),
        "frozen_seeds": {"global": 20260930, "random_partition": 20260930,
                         "clusters": 1234},
    }
    jdump(environment, os.path.join(ROOT, "SG0_ENVIRONMENT.json"))

    # ---- CAMERA_FOLDS ----
    cam_folds = {"rule": "preregistered 8 frozen cameras c0..c7; "
                         "Fold A design=c0..c3 holdout=c4..c7; "
                         "Fold B design=c4..c7 holdout=c0..c3; "
                         "offsets = cyclic rotations of design order",
                 "cells": {}}
    for cell, cd in cells.items():
        cam_folds["cells"][cell] = {
            "frozen_camera_ids": cd["frozen_camera_ids"],
            "fold_A": {"design": cd["frozen_camera_ids"][0:4],
                       "holdout": cd["frozen_camera_ids"][4:8]},
            "fold_B": {"design": cd["frozen_camera_ids"][4:8],
                       "holdout": cd["frozen_camera_ids"][0:4]},
            "cyclic_offsets": [[0, 1, 2, 3], [1, 2, 3, 0],
                               [2, 3, 0, 1], [3, 0, 1, 2]],
        }
    jdump(cam_folds, os.path.join(ROOT, "SG0_CAMERA_FOLDS.json"))

    # ---- HIERARCHY ----
    hierarchy_out = {"rule": "deterministic balanced kd-tree on theta_0 means; "
                             "split=largest range axis (tie->lowest); median split; "
                             "tie-break by Gaussian ID; leaf<=32 (L32); "
                             "L128 = ancestor two binary levels above L32",
                     "cells": {}}
    for cell, cd in cells.items():
        hierarchy_out["cells"][cell] = {
            "hierarchy": cd["hierarchy"],
            "fine_dof": cd["fine_dof"],
        }
    jdump(hierarchy_out, os.path.join(ROOT, "SG0_HIERARCHY.json"))

    # ---- BASELINE_ROLLOUTS (compact) ----
    baseline = {"cells": {}}
    for cell, cd in cells.items():
        baseline["cells"][cell] = {
            "rollouts": [
                {k: r[k] for k in ("rollout", "fold", "K", "design_positions",
                                   "holdout_positions", "L0_design", "L0_holdout",
                                   "LK_design", "LK_holdout", "base_non_descent",
                                   "train_loss_trace")}
                for r in cd["rollouts"]
            ],
        }
    jdump(baseline, os.path.join(ROOT, "SG0_BASELINE_ROLLOUTS.json"))

    # ---- ORACLE_PROJECTIONS (compact per rollout/config) ----
    oracle = {"cells": {}}
    for cell, cd in cells.items():
        oracle["cells"][cell] = {
            "configs": {
                cfg: {
                    "coarse_dof": cc["coarse_dof"],
                    "coarse_dof_ratio": cc["coarse_dof_ratio"],
                    "n_clusters": cc["n_clusters"],
                    "per_rollout": [
                        {"rollout": r["rollout"], "fold": r["fold"], "K": r["K"],
                         **{k: r.get("configs", {}).get(cfg, {})
                            for k in ("p_hold", "p_design", "oracle_improve",
                                      "LO_design", "LO_holdout", "energy_total")}}
                        for r in cd["rollouts"]
                    ],
                }
                for cfg, cc in cd["configs"].items()
            }
        }
    jdump(oracle, os.path.join(ROOT, "SG0_ORACLE_PROJECTIONS.json"))

    # ---- CONTROLS ----
    controls = {"identity_control": {}, "random_cluster_diagnostic": {}}
    for cell, cd in cells.items():
        controls["identity_control"][cell] = cd["identity_control"]
        controls["random_cluster_diagnostic"][cell] = cd["random_cluster_control"]
    controls["note"] = ("identity control is a hard gate (SG0_PROJECTION_BUG on "
                        "failure); random-cluster diagnostic is descriptive only "
                        "and does not enter the verdict")
    jdump(controls, os.path.join(ROOT, "SG0_CONTROLS.json"))

    # ---- SUMMARY (analyzer input schema) ----
    summary = {"schema_version": "SG0-v1", "cells": {}}
    for cell, cd in cells.items():
        summary["cells"][cell] = {"configs": {}}
        for cfg, cc in cd["configs"].items():
            summary["cells"][cell]["configs"][cfg] = {
                "coarse_dof_ratio_median": cc["coarse_dof_ratio"],
                "K4": cc["K4"], "K8": cc["K8"],
            }
    jdump(summary, os.path.join(ROOT, "SG0_SUMMARY.json"))
    log("wrote SG0_SUMMARY.json")

    # ---- baseline stability stop check ----
    unstable = []
    for cell, cd in cells.items():
        bnd = cd.get("base_non_descent_K8_count", 0)
        if bnd > 2:
            unstable.append((cell, bnd))

    # ---- run the analyzer (owns the verdict) ----
    if unstable:
        verdict = {"verdict": "SG0_BASELINE_UNSTABLE",
                   "unstable_cells": unstable,
                   "note": ">2/8 BASE_NON_DESENT for K=8 in a cell -> STOP"}
        jdump(verdict, os.path.join(ROOT, "SG0_VERDICT.json"))
        log("BASELINE UNSTABLE ->", unstable)
    else:
        log("running sg0_analyze.py ...")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "sg0_analyze.py"),
                            os.path.join(ROOT, "SG0_SUMMARY.json")],
                           capture_output=True, text=True)
        log("analyzer rc", r.returncode)
        print(r.stdout)
        if r.returncode != 0:
            print(r.stderr)
            raise SystemExit("analyzer failed")
        verdict = json.load(open(os.path.join(ROOT, "SG0_VERDICT.json")))

    # ---- REPORT ----
    write_report(ROOT, cells, summary, verdict, environment, unstable)
    log("wrote SG0_REPORT.md")

    # ---- MANIFEST ----
    manifest = {"campaign": "MOONSHOT0-SG0", "track": "SG0", "files": []}
    for fn in sorted(os.listdir(ROOT)):
        p = os.path.join(ROOT, fn)
        if os.path.isfile(p) and fn != "SG0_MANIFEST.json":
            manifest["files"].append({"name": fn, "bytes": os.path.getsize(p),
                                      "sha256": sha256_file(p)})
    for fn in sorted(os.listdir(CELLDIR)):
        p = os.path.join(CELLDIR, fn)
        manifest["files"].append({"name": f"cells/{fn}", "bytes": os.path.getsize(p),
                                  "sha256": sha256_file(p)})
    jdump(manifest, os.path.join(ROOT, "SG0_MANIFEST.json"))
    log("wrote SG0_MANIFEST.json")
    log("VERDICT:", verdict["verdict"])


def write_report(ROOT, cells, summary, verdict, env, unstable):
    L = []
    A = L.append
    A("# SplatMG SG0 — Future-Displacement Coarse-Space Oracle")
    A("")
    A("Frozen prereg: SG0_PREREG (2026-09-30). Oracle upper-bound study; "
      "no CUDA authoring, no 30K, no threshold tuning.")
    A("")
    A("## Environment")
    A(f"- host: {env['host']}  gpu: {env['gpu_name']}  torch: {env['torch']}")
    A(f"- renderer arm: {summary and env.get('gpu_name') and 'C0 (frozen gsplat/HIGS)'}")
    A(f"- loss: 0.8*L1 + 0.2*(1-SSIM), full resolution, no downsampling")
    A("")
    A("## Verdict")
    A(f"**{verdict['verdict']}**  (selected fixed config: "
      f"{verdict.get('selected_fixed_config')})")
    if verdict.get("strong_configs"):
        A(f"- strong configs: {verdict['strong_configs']}")
    if verdict.get("promote_configs"):
        A(f"- promote configs: {verdict['promote_configs']}")
    A("")
    A("## Per-cell, per-config metrics")
    A("")
    A("| cell | config | dof_ratio | K4 med/P10 P_hold | K8 med/P10 P_hold | "
      "K8 P_des | K8 improve | K8 non-descent |")
    A("|---|---|---|---|---|---|---|---|")
    for cell in sorted(cells):
        for cfg in ["L32/SIM13", "L32/AFF22", "L128/SIM13", "L128/AFF22"]:
            cc = cells[cell]["configs"][cfg]
            k4, k8 = cc["K4"], cc["K8"]
            A(f"| {cell} | {cfg} | {cc['coarse_dof_ratio']:.4f} | "
              f"{fmt(k4['p_hold_median'])}/{fmt(k4['p_hold_p10'])} | "
              f"{fmt(k8['p_hold_median'])}/{fmt(k8['p_hold_p10'])} | "
              f"{fmt(k8['p_design_median'])} | {k8['oracle_improve_count']}/8 | "
              f"{k8['base_non_descent_count']}/8 |")
    A("")
    A("## Hierarchy (from theta_0 means only)")
    for cell in sorted(cells):
        h = cells[cell]["hierarchy"]
        A(f"- {cell}: L32 n={h['L32']['n_clusters']} "
          f"sizes[{h['L32']['size_min']}..{h['L32']['size_max']}]; "
          f"L128 n={h['L128']['n_clusters']} "
          f"sizes[{h['L128']['size_min']}..{h['L128']['size_max']}]")
    A("")
    A("## Controls")
    for cell in sorted(cells):
        ic = cells[cell]["identity_control"]
        rc = cells[cell]["random_cluster_control"]
        A(f"- {cell}: identity max_err={ic['max_abs_err']:.2e} pass={ic['pass']}; "
          f"random-cluster K8 P_hold median={fmt(rc['K8']['p_hold_median'])} (diagnostic)")
    A("")
    A("## Interpretation")
    A(interpretation(verdict))
    A("")
    open(os.path.join(ROOT, "SG0_REPORT.md"), "w").write("\n".join(L))


def fmt(x):
    return "NA" if x is None else f"{x:.3f}"


def interpretation(verdict):
    v = verdict["verdict"]
    if v == "SG0_STRONG_PROMOTE":
        return ("A single fixed (level,basis) captures the future-displacement "
                "progress of ordinary Adam steps above the strong gates in all "
                "four cells. This justifies designing a DEPLOYABLE coarse "
                "solver in SG1 (a real V-cycle that must discover the oracle "
                "displacement without future information).")
    if v == "SG0_PROMOTE":
        return ("A single fixed (level,basis) clears the moderate promote gates "
                "in all four cells. A deployable SG1 coarse solver is justified, "
                "but the margin is thinner than the strong gate; expect SG1 to be "
                "harder.")
    if v == "SG0_BASELINE_UNSTABLE":
        return ("More than 2/8 K=8 rollouts are BASE_NON_DESCENT in a cell: the "
                "frozen baseline is not a reliable descent, so the oracle study "
                "is stopped per prereg.")
    if v == "SG0_INPUT_INVALID":
        return "Summary cell set did not match the frozen four cells; input rejected."
    return ("SG0_DROP: this specific hierarchy + low-order basis family does not "
            "contain enough future optimization progress, even with oracle future "
            "information. Per prereg this is decisive for this family; it must not "
            "be rescued by downsampling, gradient-dependent clustering, more basis "
            "modes, or per-scene basis choice.")


if __name__ == "__main__":
    main()

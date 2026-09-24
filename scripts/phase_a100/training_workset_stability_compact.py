#!/usr/bin/env python3
"""Canonical A100 gsplat-1.5.3 workset-stability audit with compact exact tensors.

No renderer changes are made.  The script observes gsplat's rasterization info
and computes exact Jaccard overlap online for lags 1/2/4/8 without materializing
Python sets of Gaussian-tile memberships.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "src"))
from gsplat import rasterization
from benchmark_framework import load_ply
from scripts.epic05.phase7.dataset import GTDataset
from scripts.epic05.phase7.gaussian_model import GaussianModel
from scripts.epic05.phase7.loss import combined_loss

LAGS = (1, 2, 4, 8)


def worksets(info: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sorted unique exact integer representations of A/B/C on GPU.

    A = original Gaussian IDs that survived projection.
    B = active tile IDs.
    C = encoded (original Gaussian ID, tile ID) pairs.

    gsplat 1.5.3 semantics were verified independently: flatten_ids indexes
    gaussian_ids; isect_ids is a packed sorting key, not a Gaussian-ID list.
    """
    gids = info["gaussian_ids"].long()
    flat = info["flatten_ids"].long()
    starts = info["isect_offsets"][0].reshape(-1).long()
    n_entries = flat.numel()
    n_tiles = starts.numel()
    ends = torch.cat((starts[1:], starts.new_tensor([n_entries])))
    counts = ends - starts
    active = torch.nonzero(counts > 0, as_tuple=False).flatten()
    entry_tiles = torch.repeat_interleave(
        torch.arange(n_tiles, dtype=torch.long, device=gids.device), counts
    )
    # Original gaussian IDs can be up to ~1.6M, tile count < 2^14 here;
    # this encoding is collision-free in signed int64.
    members = torch.sort(gids[flat] * n_tiles + entry_tiles).values
    return torch.sort(gids).values, active, members


def jac_sorted(a: torch.Tensor, b: torch.Tensor) -> float:
    """Exact Jaccard overlap for sorted, unique integer tensors."""
    if a.numel() == 0 and b.numel() == 0:
        return 1.0
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    a = torch.unique_consecutive(a)
    b = torch.unique_consecutive(b)
    p = torch.searchsorted(b, a)
    valid = p < b.numel()
    equal = torch.zeros_like(valid)
    equal[valid] = b[p[valid]] == a[valid]
    common = int(equal.sum().item())
    return common / (a.numel() + b.numel() - common)


def descriptive(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    x = np.asarray(values, dtype=np.float64)
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std()),
            "p10": float(np.percentile(x, 10)), "p50": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90)), "min": float(x.min()), "max": float(x.max())}


def renderer(model: GaussianModel, camera, tile_size: int):
    d = model.forward()
    image, alpha, info = rasterization(
        means=d["xyz"], quats=d["rotations"], scales=d["scales"],
        opacities=d["opacity"], colors=d["shs"],
        viewmats=camera.viewmatrix.unsqueeze(0), Ks=camera.K.unsqueeze(0),
        width=camera.image_width, height=camera.image_height,
        tile_size=tile_size, packed=True, sh_degree=model.sh_degree,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB", sparse_grad=False,
        absgrad=False,
    )
    return image[0].clamp(0, 1), info


def main(args):
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("BLOCKED: canonical A100 CUDA execution is unavailable")
    import gsplat
    if gsplat.__version__ != "1.5.3":
        raise RuntimeError(f"BLOCKED: expected gsplat 1.5.3, got {gsplat.__version__}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = ROOT / "results" / "phase-a100"; out.mkdir(parents=True, exist_ok=True)
    reports = ROOT / "reports" / "phase-a100"; reports.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    ds = GTDataset(args.scene, ROOT, resolution=args.resolution, device=device)
    sfm = load_ply(str(ROOT / "data" / "official" / "mipnerf360" / args.scene / "point_cloud.ply"), device=device)
    model = GaussianModel(sfm["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=device)
    model.init_from_sfm(sfm["xyz"], torch.logit(torch.full((sfm["xyz"].shape[0], 1), .1, device=device)),
                        sfm["scales"], sfm["rotations"], sfm["shs"])
    extent = float(sfm["xyz"].norm(dim=-1).max().item())

    def make_optim():
        return torch.optim.Adam([
            {"params":[model.xyz], "lr":1.6e-4 * extent, "eps":1e-15},
            {"params":[model.rotations], "lr":1e-3, "eps":1e-15},
            {"params":[model.scales], "lr":5e-3, "eps":1e-15},
            {"params":[model.opacity], "lr":5e-2, "eps":1e-15},
            {"params":[model.shs], "lr":2.5e-3, "eps":1e-15},
        ])
    optim = make_optim()

    histories = {k: deque(maxlen=max(LAGS)+1) for k in "ABC"}
    overlap = {k: {lag: [] for lag in LAGS} for k in "ABC"}
    phase = {p: {k: {lag: [] for lag in LAGS} for k in "ABC"}
             for p in ("early", "middle", "late")}
    event = {"densification": {k: [] for k in "ABC"}, "pruning": {k: [] for k in "ABC"}}
    rows, dens_steps, prune_steps = [], [], []
    eval_prev = {}  # camera index -> (A,B,C); retained on GPU only
    same_view = {k: [] for k in "ABC"}
    cross_view_a = []
    t0 = time.perf_counter()

    for step in range(args.steps):
        camera, target = ds.get_item(step % len(ds))
        pred, info = renderer(model, camera, args.tile_size)
        a, b, c = worksets(info)
        current = {"A": a, "B": b, "C": c}
        category = "early" if step < 100 else ("middle" if step < 400 else "late")
        step_ovs = {}
        for name, values in current.items():
            hist = histories[name]
            for lag in LAGS:
                if len(hist) >= lag:
                    v = jac_sorted(values, hist[-lag])
                    overlap[name][lag].append(v)
                    phase[category][name][lag].append(v)
                    if lag == 1: step_ovs[name] = v
            hist.append(values.detach())

        loss_data = combined_loss(pred, target, lambda_dssim=.2)
        loss = loss_data["loss"]
        mse = torch.mean((pred - target) ** 2).item()
        psnr = 10.0 * math.log10(1.0 / max(mse, 1e-10))
        optim.zero_grad(set_to_none=True); loss.backward(); model.accumulate_positional_gradient(); optim.step()

        did_dens = did_prune = False
        if step >= 100 and step < 15000 and step % 100 == 0:
            e = model.densification(grad_threshold=2e-4, clone_max_screen_size=100., split_max_screen_size=100.)
            did_dens = (e["cloned"] + e["split"]) > 0
            if did_dens: dens_steps.append(step)
        if step >= 100 and step % 100 == 0:
            n = model.prune_and_reset(opacity_threshold=.005, reset_interval=3000, current_step=step)
            did_prune = n > 0
            if did_prune: prune_steps.append(step)
        if did_dens or did_prune:
            for name in "ABC":
                if name in step_ovs:
                    event["densification" if did_dens else "pruning"][name].append(step_ovs[name])
            optim = make_optim()

        # Same-viewpoint and cross-view measurement, evaluated every 50 steps.
        if step % 50 == 0 or step == args.steps - 1:
            current_eval_a = []
            with torch.no_grad():
                for ci in range(min(5, len(ds))):
                    ec, _ = ds.get_item(ci)
                    _, ei = renderer(model, ec, args.tile_size)
                    ea, eb, ecset = worksets(ei)
                    values = {"A":ea, "B":eb, "C":ecset}
                    if ci in eval_prev:
                        for name in "ABC": same_view[name].append(jac_sorted(values[name], eval_prev[ci][name]))
                    eval_prev[ci] = {name: value.detach() for name, value in values.items()}
                    current_eval_a.append(ea)
            for i in range(len(current_eval_a)):
                for j in range(i+1, len(current_eval_a)):
                    cross_view_a.append(jac_sorted(current_eval_a[i], current_eval_a[j]))

        rows.append({"iteration":step, "n_gaussian":int(model.xyz.shape[0]), "n_visible":int(a.numel()),
                     "n_tiles":int(b.numel()), "n_memberships":int(c.numel()), "loss":float(loss.item()),
                     "psnr":psnr, "vg_jaccard_lag1":step_ovs.get("A"),
                     "at_jaccard_lag1":step_ovs.get("B"), "mb_jaccard_lag1":step_ovs.get("C")})
        if step % 25 == 0 or step == args.steps-1:
            print(f"step={step:04d} loss={loss.item():.5f} psnr={psnr:.2f} N={model.xyz.shape[0]:,} "
                  f"A={a.numel():,} B={b.numel():,} C={c.numel():,} "
                  f"J1=({step_ovs.get('A',-1):.4f},{step_ovs.get('B',-1):.4f},{step_ovs.get('C',-1):.4f})", flush=True)

    stats = {name: {f"lag_{lag}":descriptive(overlap[name][lag]) for lag in LAGS} for name in "ABC"}
    pstats = {p:{name:{f"lag_{lag}":descriptive(phase[p][name][lag]) for lag in LAGS} for name in "ABC"} for p in phase}
    sv = {name:descriptive(same_view[name]) for name in "ABC"}
    manifest = {
        "timestamp_utc":datetime.now(timezone.utc).isoformat(), "hostname":os.uname().nodename,
        "gpu":torch.cuda.get_device_name(0), "gpu_count":torch.cuda.device_count(),
        "torch":torch.__version__, "cuda_runtime":torch.version.cuda, "gsplat":gsplat.__version__,
        "scene":args.scene, "steps":args.steps, "resolution":args.resolution, "tile_size":args.tile_size,
        "renderer_change":False, "instrumentation_only":True,
    }
    result = {"meta":{"type":"CANONICAL_REPLICATION","workset_c_encoding":"gaussian_id * tile_count + tile_id"},
              "manifest":manifest, "training":{"duration_s":time.perf_counter()-t0, "initial_n":int(sfm["xyz"].shape[0]),
              "final_n":int(model.xyz.shape[0]), "loss_initial":rows[0]["loss"], "loss_final":rows[-1]["loss"],
              "psnr_initial":rows[0]["psnr"], "psnr_final":rows[-1]["psnr"], "nan_or_inf":not all(math.isfinite(r["loss"]) for r in rows)},
              "worksets":{"A":"info.gaussian_ids","B":"nonempty info.isect_offsets intervals","C":"exact Gaussian-tile membership from gaussian_ids[flatten_ids] and tile interval"},
              "training_view_overlap":stats, "phase_overlap":pstats,
              "same_viewpoint_overlap":sv, "cross_view_visible_gaussian":descriptive(cross_view_a),
              "event_conditioned_lag1":{kind:{name:descriptive(vals) for name,vals in d.items()} for kind,d in event.items()},
              "events":{"densification_steps":dens_steps,"pruning_steps":prune_steps}}
    (out / "training_workset_stability_replication_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "training_workset_stability_replication.json").write_text(json.dumps(result, indent=2))
    with (out / "training_workset_overlap_series_replication.csv").open("w", newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    def p(name): return stats[name]["lag_1"].get("p50",float('nan'))
    report = f"# Training Workset Stability Replication — A100 Canonical\n\n"
    report += f"- Host: `{manifest['hostname']}`; GPU: `{manifest['gpu']}`; gsplat: `{manifest['gsplat']}`\n"
    report += f"- Scene `{args.scene}`, {args.steps} steps, tile {args.tile_size}, {args.resolution}\n"
    report += f"- Training: PSNR {result['training']['psnr_initial']:.2f} → {result['training']['psnr_final']:.2f}; N {result['training']['initial_n']:,} → {result['training']['final_n']:,}; NaN/Inf={result['training']['nan_or_inf']}\n\n"
    report += "## Exact online overlap (training view; camera changes confound this metric)\n\n"
    for name in "ABC": report += f"- {name}, lag-1: {json.dumps(stats[name]['lag_1'])}\n"
    report += "\n## Same-viewpoint stability (five fixed cameras; consecutive evaluation checkpoints)\n\n"
    for name in "ABC": report += f"- {name}: {json.dumps(sv[name])}\n"
    report += "\n## Evidence classification\n\n"
    report += f"- Gaussian-level reuse (A): `{'SUPPORTED' if sv['A'].get('p50',0)>=.90 else 'PARTIALLY_SUPPORTED' if sv['A'].get('p50',0)>=.70 else 'FALSIFIED'}`\n"
    report += f"- Tile-level reuse (B): `{'SUPPORTED' if sv['B'].get('p50',0)>=.90 else 'PARTIALLY_SUPPORTED' if sv['B'].get('p50',0)>=.65 else 'FALSIFIED'}`\n"
    report += "- Gaussian-level cache + event-aware tile rebuild: `HYPOTHESIS` (overlap does not establish speedup).\n"
    (reports / "training_workset_stability_replication.md").write_text(report)
    print(json.dumps({"A_lag1":stats['A']['lag_1'],"B_lag1":stats['B']['lag_1'],"C_lag1":stats['C']['lag_1'],"same_view":sv}, indent=2), flush=True)

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--scene",default="room"); ap.add_argument("--steps",type=int,default=500)
    ap.add_argument("--tile-size",type=int,default=16); ap.add_argument("--resolution",default="1080p")
    ap.add_argument("--seed",type=int,default=42)
    main(ap.parse_args())

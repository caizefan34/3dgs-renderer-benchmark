#!/usr/bin/env python3
"""
Phase 9A — M2 Gradient Correctness: packed vs dense gradient comparison.
Quick check: compare gradient norms between packed and dense modes.
"""

import json, math, sys, time
from datetime import datetime, timezone
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark_framework import load_ply, load_cameras_from_json
from gsplat import rasterization

device = "cuda"
scene = "room"

# Load data
data = load_ply(str(REPO_ROOT / "data" / "official" / "mipnerf360" / scene / "point_cloud.ply"), device=device)
x = data["xyz"].requires_grad_(True)
q = torch.nn.functional.normalize(data["rotations"], dim=-1).requires_grad_(True)
s = torch.exp(data["scales"]).requires_grad_(True)
o = torch.sigmoid(data["opacity"]).squeeze(-1).requires_grad_(True)
sh = data["shs"].requires_grad_(True)

cam = load_cameras_from_json(str(REPO_ROOT / "data" / "official" / "mipnerf360" / scene / "cameras.json"))[0]
vm = cam.viewmatrix.unsqueeze(0).to(device)
K = cam.K.unsqueeze(0).to(device)
H, W = cam.image_height, cam.image_width

results = {}

for packed, label in [(True, "packed"), (False, "dense")]:
    print(f"\n{'='*60}")
    print(f"  Gradient check: {label}")
    print(f"{'='*60}")

    # Recreate params
    x = data["xyz"].clone().detach().requires_grad_(True)
    q = torch.nn.functional.normalize(data["rotations"].clone().detach(), dim=-1).requires_grad_(True)
    s = torch.exp(data["scales"].clone().detach()).requires_grad_(True)
    o = torch.sigmoid(data["opacity"].clone().detach()).squeeze(-1).requires_grad_(True)
    sh = data["shs"].clone().detach().requires_grad_(True)

    rendered, _, _ = rasterization(
        means=x, quats=q, scales=s, opacities=o, colors=sh,
        viewmats=vm, Ks=K, width=W, height=H,
        tile_size=16, packed=packed, sh_degree=3, render_mode="RGB")

    loss = rendered.sum()
    loss.backward()

    grads = {}
    for name, p in [("means", x), ("quats", q), ("scales", s), ("opacity", o), ("shs", sh)]:
        g = p.grad
        gn = g.norm().item() if g is not None else 0.0
        fin = bool(torch.isfinite(g).all()) if g is not None else True
        nan = bool(torch.isnan(g).any()) if g is not None else False
        inf = bool(torch.isinf(g).any()) if g is not None else False
        grads[name] = {"norm": gn, "finite": fin, "nan": nan, "inf": inf, "shape": list(g.shape) if g is not None else None}
        print(f"    {name:8s}: norm={gn:.6e} finite={fin} nan={nan} inf={inf}")

    results[label] = grads
    del x, q, s, o, sh, rendered
    torch.cuda.empty_cache()

# Compare
print(f"\n{'='*60}")
print(f"  Gradient comparison: packed vs dense")
print(f"{'='*60}")
comparisons = {}
all_pass = True
for key in results["packed"]:
    n_p = results["packed"][key]["norm"]
    n_d = results["dense"][key]["norm"]
    rel_diff = abs(n_p - n_d) / max(max(abs(n_p), abs(n_d)), 1e-10)
    comparisons[key] = {"packed_norm": n_p, "dense_norm": n_d, "rel_diff": rel_diff}
    ok = rel_diff < 1e-5 and results["packed"][key]["finite"] and results["dense"][key]["finite"]
    all_pass = all_pass and ok
    print(f"    {key:8s}: packed={n_p:.6e} dense={n_d:.6e} rel_diff={rel_diff:.2e} {'OK' if ok else 'FAIL'}")

results["comparison"] = comparisons
results["verdict"] = "PASS" if all_pass else "FAIL"

out_dir = REPO_ROOT / "results" / "epic05" / "phase9a"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"m2_gradient_{scene}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n  Verdict: {'PASS' if all_pass else 'FAIL'}")
print(f"  Saved: {out_path}")

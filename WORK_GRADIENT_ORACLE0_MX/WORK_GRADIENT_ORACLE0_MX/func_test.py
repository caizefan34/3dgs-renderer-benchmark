#!/usr/bin/env python3
"""Minimal real functional test: call rasterize_gaussian_higs_dynamic on a
tiny synthetic state, full vs sparse (tile_mask). Checks the native op
actually runs and whether tile_mask changes the isect workload."""
import torch, math

torch.manual_seed(0)
dev = "cuda:0"
C, N = 1, 2000
H, W = 64, 64
tile = 16
th, tw = math.ceil(H/tile), math.ceil(W/tile)

means = (torch.rand(C, N, 3, device=dev) - 0.5) * 3
means[..., 2] += 5.0  # in front
quats = torch.zeros(C, N, 4, device=dev); quats[..., 0] = 1
scales = torch.full((C, N, 3), -3.0, device=dev)
opac = torch.zeros(C, N, 1, device=dev); opac[..., 0] = 2.0
colors = torch.rand(C, N, 3, device=dev)
# viewmat looking down +z
viewmats = torch.eye(4, device=dev).unsqueeze(0)
viewmats[..., 2, 3] = -5.0
f = 50.0
Ks = torch.tensor([[[f,0,W/2],[0,f,H/2],[0,0,1]]], device=dev, dtype=torch.float32)

from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as R

def run(tag, tile_mask):
    try:
        out = R(means, quats, scales, opac, colors, sh_degree=0,
                viewmats=viewmats, Ks=Ks, width=W, height=H, tile_size=tile,
                tile_mask=tile_mask, enable_culling=False, use_higs_culling=False)
        frame = out["frame"]
        meta = out.get("metadata", {})
        n_isects = meta.get("n_isects", None)
        print(f"[{tag}] OK frame={tuple(frame.shape)} finite={torch.isfinite(frame).all().item()} n_isects={n_isects}")
        return True, n_isects
    except Exception as e:
        import traceback
        print(f"[{tag}] FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False, None

# FULL
ok_full, ni_full = run("full", None)

# SPARSE: select 25% of tiles
mask = torch.zeros((C, th, tw), dtype=torch.bool, device=dev)
mask.reshape(C, -1)[:, ::4] = True
ok_sparse, ni_sparse = run("sparse_mask", mask)

if ok_full and ok_sparse:
    print("\nSPARSE WORKLOAD REDUCTION:", f"n_isects full={ni_full} sparse={ni_sparse}")

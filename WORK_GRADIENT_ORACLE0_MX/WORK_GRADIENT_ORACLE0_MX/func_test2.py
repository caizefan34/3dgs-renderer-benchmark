#!/usr/bin/env python3
"""Gate D/E: full vs sparse forward+backward with correct packed layout.
means (N,3), quats (N,4), scales (N,3), opac (N,1), colors (N,K,3),
viewmats (C,4,4), Ks (C,3,3), tile_mask (C,th,tw)."""
import torch, math

torch.manual_seed(0)
dev = "cuda:0"
C, N = 1, 3000
H, W = 64, 64
tile = 16
K = 3  # sh_degree=0 -> but colors still [N,1,3]?
th, tw = math.ceil(H/tile), math.ceil(W/tile)

means = (torch.rand(N, 3, device=dev) - 0.5) * 3
means[:, 2] += 5.0
means.requires_grad_(True)
quats = torch.zeros(N, 4, device=dev); quats[:, 0] = 1.0; quats.requires_grad_(True)
scales = torch.full((N, 3), -3.0, device=dev); scales.requires_grad_(True)
opac = torch.full((N, 1), 2.0, device=dev); opac.requires_grad_(True)
colors = torch.rand(N, 1, 3, device=dev); colors.requires_grad_(True)  # [N, K, 3]

viewmats = torch.eye(4, device=dev).unsqueeze(0).repeat(C,1,1)
viewmats[:, 2, 3] = -5.0
f = 50.0
Ks = torch.tensor([[[f,0,W/2],[0,f,H/2],[0,0,1]]], device=dev, dtype=torch.float32).repeat(C,1,1)

from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as R

def run(tag, tile_mask, want_grad=True):
    params = [means, quats, scales, opac, colors]
    for p in params: p.grad = None
    try:
        out = R(means, quats, scales, opac, colors, sh_degree=0,
                viewmats=viewmats, Ks=Ks, width=W, height=H, tile_size=tile,
                tile_mask=tile_mask, enable_culling=False, use_higs_culling=False)
        frame = out["frame"]  # [C,H,W,3]
        meta = out.get("metadata", {}) or {}
        n_isects = meta.get("n_isects", None)
        sampled = meta.get("sampled_tile_ratio", None)
        loss = frame.mean()
        if want_grad:
            loss.backward()
            gshapes = {n: (p.grad.shape if p.grad is not None else None) for n,p in
                       zip(["means","quats","scales","opac","colors"], params)}
            gfinite = all((p.grad is None) or torch.isfinite(p.grad).all().item() for p in params)
        else:
            gshapes, gfinite = None, None
        print(f"[{tag}] OK frame={tuple(frame.shape)} finite={torch.isfinite(frame).all().item()} "
              f"loss={loss.item():.4f} n_isects={n_isects} sampled_ratio={sampled} grad_finite={gfinite}")
        print(f"       grad_shapes={gshapes}")
        return True, n_isects
    except Exception as e:
        import traceback
        print(f"[{tag}] FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False, None

ok_full, ni_full = run("full", None)
mask = torch.zeros((C, th, tw), dtype=torch.bool, device=dev)
mask.reshape(C, -1)[:, ::4] = True
ok_sparse, ni_sparse = run("sparse_mask", mask)

if ok_full and ok_sparse:
    print("\n=== WORKLOAD CHECK ===")
    print(f"n_isects full={ni_full} sparse={ni_sparse}")
    if ni_full and ni_sparse:
        print(f"reduction={1-ni_sparse/ni_full:.2%} (expect ~75% fewer for 25% tiles)")

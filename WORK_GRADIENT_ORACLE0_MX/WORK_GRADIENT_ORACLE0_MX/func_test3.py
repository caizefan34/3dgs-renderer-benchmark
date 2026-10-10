#!/usr/bin/env python3
"""Gate D/E/F: full vs sparse forward+backward, real sparse kernel.
Correct HiGS layout: means(N,3) quats(N,4) scales(N,3) opac(N,) colors(N,K,3)."""
import torch, math

torch.manual_seed(0)
dev = "cuda:0"
C, N = 1, 3000
H, W = 64, 64
tile = 16
th, tw = math.ceil(H/tile), math.ceil(W/tile)
sh_degree = 0
K = (sh_degree+1)**2

means = ((torch.rand(N, 3, device=dev) - 0.5) * 3).contiguous()
means[:, 2] += 5.0
quats = torch.zeros(N, 4, device=dev); quats[:, 0] = 1.0
scales = torch.full((N, 3), -3.0, device=dev)
opac = torch.full((N,), 2.0, device=dev)          # 1D
colors = torch.rand(N, K, 3, device=dev)          # [N,K,3]
for p in (means, quats, scales, opac, colors):
    p.requires_grad_(True)

viewmats = torch.eye(4, device=dev).unsqueeze(0).repeat(C,1,1)
viewmats[:, 2, 3] = -5.0
f = 50.0
Ks = torch.tensor([[[f,0,W/2],[0,f,H/2],[0,0,1]]], device=dev, dtype=torch.float32).repeat(C,1,1)

from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as R

def run(tag, tile_mask):
    for p in (means, quats, scales, opac, colors):
        if p.grad is not None: p.grad = None
    try:
        out = R(means, quats, scales, opac, colors, sh_degree=sh_degree,
                viewmats=viewmats, Ks=Ks, width=W, height=H, tile_size=tile,
                tile_mask=tile_mask, enable_culling=False, use_higs_culling=False)
        frame = out["frame"]  # [C,H,W,3]
        meta = out.get("metadata", {}) or {}
        n_isects = meta.get("n_isects", None)
        sampled = meta.get("sampled_tile_ratio", None)
        torch.cuda.synchronize()
        loss = frame.mean()
        loss.backward()
        torch.cuda.synchronize()
        gnames = ["means","quats","scales","opac","colors"]
        gshapes = {n: (tuple(p.grad.shape) if p.grad is not None else None)
                   for n,p in zip(gnames,(means,quats,scales,opac,colors))}
        gfinite = all((p.grad is None) or torch.isfinite(p.grad).all().item()
                      for p in (means,quats,scales,opac,colors))
        gnz = {n: int((p.grad.abs()>0).sum().item()) if p.grad is not None else 0
               for n,p in zip(gnames,(means,quats,scales,opac,colors))}
        print(f"[{tag}] OK frame={tuple(frame.shape)} finite={torch.isfinite(frame).all().item()} "
              f"loss={loss.item():.4f} n_isects={n_isects} sampled_ratio={sampled}")
        print(f"       grad_finite={gfinite} grad_nonzero={gnz}")
        return True, n_isects, gshapes
    except Exception as e:
        import traceback
        print(f"[{tag}] FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False, None, None

ok_full, ni_full, gs_full = run("full", None)
mask = torch.zeros((C, th, tw), dtype=torch.bool, device=dev)
mask.reshape(C, -1)[:, ::4] = True   # select 25% of tiles
ok_sparse, ni_sparse, gs_sparse = run("sparse_mask", mask)

print("\n=== GATE E: sparse workload reduction ===")
if ok_full and ok_sparse:
    print(f"n_isects full={ni_full} sparse={ni_sparse}")
    if ni_full and ni_sparse:
        print(f"reduction={1-ni_sparse/ni_full:.2%}")
    # gradient nonzero should also reduce for sparse
    print("grad_nonzero full  :", gs_full and {k:v for k,v in []} )

#!/usr/bin/env python3
"""AccuTile A100 smoke test: minimal OFF/ON forward+backward parity check.
Ported from accutile_smoke.py for gsplat v1.5.3 API.
"""
import sys, json, torch
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/gsplat-accutile-v153")
import gsplat
from gsplat import rasterization

def main():
    print(f"gsplat version: {gsplat.__version__}")
    print(f"gsplat file: {gsplat.__file__}")
    torch.manual_seed(0)
    n = 128
    device = "cuda"
    means = torch.randn(n, 3, device=device, requires_grad=True)
    means.data[:, 2].abs_().add_(2)
    quats = torch.randn(n, 4, device=device, requires_grad=True)
    quats.data /= quats.data.norm(dim=-1, keepdim=True)
    scales = (torch.rand(n, 3, device=device) * 0.1 + 0.01).requires_grad_()
    opacities = (torch.rand(n, device=device) * 0.8 + 0.1).requires_grad_()
    colors = torch.rand(n, 3, device=device, requires_grad=True)
    viewmats = torch.eye(4, device=device)[None]
    Ks = torch.tensor([[100., 0., 64.], [0., 100., 64.], [0., 0., 1.]], device=device)[None]
    inputs = (means, quats, scales, opacities, colors)
    runs = []
    for enabled in (False, True):
        for tensor in inputs:
            tensor.grad = None
        rgb, alpha, meta = rasterization(
            means, quats, scales, opacities, colors, viewmats, Ks, 128, 128,
            packed=False, accutile=enabled,
        )
        (rgb.sum() + alpha.sum()).backward()
        torch.cuda.synchronize()
        runs.append((rgb.detach(), alpha.detach(), int(meta["tiles_per_gauss"].sum()),
                     [tensor.grad.detach().clone() for tensor in inputs]))
    result = {
        "gsplat_version": gsplat.__version__,
        "gsplat_file": gsplat.__file__,
        "intersections_off": runs[0][2],
        "intersections_on": runs[1][2],
        "rgb_max_abs": float((runs[0][0] - runs[1][0]).abs().max()),
        "alpha_max_abs": float((runs[0][1] - runs[1][1]).abs().max()),
        "rgb_mean_abs": float((runs[0][0] - runs[1][0]).abs().mean()),
        "alpha_mean_abs": float((runs[0][1] - runs[1][1]).abs().mean()),
        "gradients": {},
    }
    grad_names = ("means", "quats", "scales", "opacities", "colors")
    for name, a, b in zip(grad_names, runs[0][3], runs[1][3]):
        diff = (a - b).abs()
        result["gradients"][name] = {
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
        }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

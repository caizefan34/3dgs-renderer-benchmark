"""Minimal OFF/ON forward-and-backward parity check for gsplat AccuTile."""

import torch
from gsplat import rasterization


def main():
    torch.manual_seed(0)
    n = 128
    device = "cuda"
    means = torch.randn(n, 3, device=device, requires_grad=True)
    means.data[:, 2].abs_().add_(2)
    quats = torch.randn(n, 4, device=device, requires_grad=True)
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
    print("intersections", runs[0][2], runs[1][2])
    for name, a, b in (("rgb", runs[0][0], runs[1][0]), ("alpha", runs[0][1], runs[1][1])):
        print(name, "max_abs", float((a - b).abs().max()))
    for name, a, b in zip(("means", "quats", "scales", "opacities", "colors"), runs[0][3], runs[1][3]):
        print(name, "grad_max_abs", float((a - b).abs().max()))


if __name__ == "__main__":
    main()

import torch
from gsplat.experimental.render.functional.gaussian_inference import _higs_dynamic_forward

torch.manual_seed(4)
dev = "cuda"
n = 256
means = (torch.randn(n, 3, device=dev) * 0.2).requires_grad_()
means.data[:, 2] += 3.0
quats = torch.nn.functional.normalize(torch.randn(n, 4, device=dev), dim=-1).requires_grad_()
scales = (torch.rand(n, 3, device=dev) * 0.04 + .01).requires_grad_()
opacities = (torch.rand(n, device=dev) * .5 + .25).requires_grad_()
sh = (torch.randn(n, 16, 3, device=dev) * .1).requires_grad_()
vm = torch.eye(4, device=dev)[None, None]
K = torch.tensor([[64., 0., 31.5], [0., 64., 31.5], [0., 0., 1.]], device=dev)[None, None]
out = _higs_dynamic_forward(means, quats, scales, opacities, sh, viewmats=vm, Ks=K,
    width=64, height=64, sh_degree=3, backward_mode="higs_native", enable_culling=True)
(out["frame"].square().mean() + out["alpha"].mean()).backward()
print(out["metadata"])
print(*(float(x.grad.abs().sum()) for x in (means, quats, scales, opacities, sh)))

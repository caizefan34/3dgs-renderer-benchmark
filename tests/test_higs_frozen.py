import pytest
import torch

device = torch.device("cuda:0")
_skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")

def _make_gaussians(N, device, seed=42):
    torch.manual_seed(seed)
    means = torch.randn(N, 3, device=device)
    means[:, 2] = means[:, 2].abs() + 2.0
    quats = torch.randn(N, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.rand(N, 3, device=device) * 0.1 + 0.01
    opacities = torch.rand(N, device=device) * 0.8 + 0.1
    colors = torch.sigmoid(torch.randn(N, 3, device=device))
    return means, quats, scales, opacities, colors

def _make_camera(device, width=128, height=96):
    K = torch.tensor([[128.0, 0.0, width/2.0], [0.0, 128.0, height/2.0], [0.0, 0.0, 1.0]], device=device)
    viewmat = torch.eye(4, device=device)
    return viewmat, K, width, height

class TestStageBFrozen:
    @_skip_no_cuda
    def test_frozen_function_importable(self):
        from gsplat.experimental import rasterize_gaussian_higs_frozen
        assert callable(rasterize_gaussian_higs_frozen)

    @_skip_no_cuda
    def test_frozen_gradients_all_params(self):
        from gsplat.experimental import rasterize_gaussian_higs_frozen
        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        viewmat, K, w, h = _make_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)
        result = rasterize_gaussian_higs_frozen(
            means, quats, scales, opacities, colors,
            viewmats=viewmats, Ks=Ks, width=w, height=h,
        )
        result["frame"].sum().backward()
        for name, t in [("means", means), ("quats", quats), ("scales", scales),
                         ("opacities", opacities), ("colors", colors)]:
            assert t.grad is not None, f"{name}.grad is None"
            assert t.grad.isfinite().all(), f"{name}.grad has non-finite values"
            assert t.grad.abs().sum() > 0, f"{name}.grad is all zeros"

    @_skip_no_cuda
    def test_frozen_forward_aligned_with_stage_a(self):
        from gsplat.experimental import rasterize_gaussian_higs_frozen, rasterize_gaussian_higs_trainable
        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        viewmat, K, w, h = _make_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)
        result_b = rasterize_gaussian_higs_frozen(
            means, quats, scales, opacities, colors,
            viewmats=viewmats, Ks=Ks, width=w, height=h,
        )
        result_a = rasterize_gaussian_higs_trainable(
            means, quats, scales, opacities, colors,
            viewmats=viewmats, Ks=Ks, width=w, height=h,
            differentiable=True, sh_degree=None,
        )
        torch.testing.assert_close(result_b["frame"], result_a.frame, atol=1e-5, rtol=1e-5)
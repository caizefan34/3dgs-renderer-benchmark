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


class TestFreezeTopology:
    """Test freeze_topology parameter validation."""

    @_skip_no_cuda
    def test_freeze_topology_accepts_same_count(self):
        """freeze_topology=True should work with consistent Gaussian count."""
        from gsplat.experimental import rasterize_gaussian_higs_frozen

        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        viewmat, K, w, h = _make_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)

        r1 = rasterize_gaussian_higs_frozen(
            means, quats, scales, opacities, colors,
            viewmats=viewmats, Ks=Ks, width=w, height=h,
            freeze_topology=True,
        )
        assert r1["metadata"]["freeze_topology"] == True

        r2 = rasterize_gaussian_higs_frozen(
            means, quats, scales, opacities, colors,
            viewmats=viewmats, Ks=Ks, width=w, height=h,
            freeze_topology=True,
        )
        assert r2["metadata"]["render_count"] > r1["metadata"]["render_count"]

    @_skip_no_cuda
    def test_freeze_topology_rejects_count_change(self):
        """freeze_topology=True should raise if Gaussian count changes."""
        from gsplat.experimental import rasterize_gaussian_higs_frozen

        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        viewmat, K, w, h = _make_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)

        rasterize_gaussian_higs_frozen(
            means, quats, scales, opacities, colors,
            viewmats=viewmats, Ks=Ks, width=w, height=h,
            freeze_topology=True,
        )

        means2, quats2, scales2, opacities2, colors2 = _make_gaussians(N + 10, device)
        with pytest.raises(RuntimeError, match="freeze_topology=True"):
            rasterize_gaussian_higs_frozen(
                means2, quats2, scales2, opacities2, colors2,
                viewmats=viewmats, Ks=Ks, width=w, height=h,
                freeze_topology=True,
            )

    @_skip_no_cuda
    def test_freeze_topology_false_allows_change(self):
        """freeze_topology=False should allow Gaussian count to change."""
        from gsplat.experimental import rasterize_gaussian_higs_frozen

        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        viewmat, K, w, h = _make_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)

        rasterize_gaussian_higs_frozen(
            means, quats, scales, opacities, colors,
            viewmats=viewmats, Ks=Ks, width=w, height=h,
            freeze_topology=False,
        )

        means2, quats2, scales2, opacities2, colors2 = _make_gaussians(N + 10, device)
        result = rasterize_gaussian_higs_frozen(
            means2, quats2, scales2, opacities2, colors2,
            viewmats=viewmats, Ks=Ks, width=w, height=h,
            freeze_topology=False,
        )
        assert result["metadata"]["n_gaussians"] == N + 10

class TestHiGSCulling:
    """Test HiGS-native culling via get_visible_mask."""


    @_skip_no_cuda
    def test_higs_culling_function_importable(self):
        from gsplat.experimental.render.functional.gaussian_inference import (
            _cull_gaussians_higs,
        )
        assert callable(_cull_gaussians_higs)

    @_skip_no_cuda
    def test_higs_culling_differentiable_pipeline(self):
        """Verify culling-based pipeline with HiGS culling is differentiable."""
        from gsplat.experimental.render.functional.gaussian_inference import (
            _cull_gaussians_higs,
        )
        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        viewmat, K, w, h = _make_camera(device)

        visible_ids, visible_mask, ratio = _cull_gaussians_higs(
            means, quats, scales, opacities, colors,
            viewmat, K, w, h, None,
        )
        assert ratio >= 0.0
        if visible_ids.numel() > 0:
            from gsplat.rendering import rasterization
            viewmats = viewmat.unsqueeze(0).unsqueeze(0)
            Ks = K.unsqueeze(0).unsqueeze(0)
            m = means[visible_ids].unsqueeze(0)
            q = quats[visible_ids].unsqueeze(0)
            s = scales[visible_ids].unsqueeze(0)
            o = opacities[visible_ids].unsqueeze(0)
            c = colors[visible_ids].unsqueeze(0)
            renders, alphas, _ = rasterization(
                means=m, quats=q, scales=s, opacities=o, colors=c,
                viewmats=viewmats, Ks=Ks, width=w, height=h, packed=True,
            )
            renders.sum().backward()
            for name, t in [("means", means), ("quats", quats), ("scales", scales),
                             ("opacities", opacities), ("colors", colors)]:
                assert t.grad is not None, f"{name}.grad is None"
                assert t.grad.isfinite().all(), f"{name}.grad has non-finite values"
            if (~visible_mask).any():
                assert means.grad[~visible_mask].abs().sum() == 0

    @_skip_no_cuda
    def test_higs_culling_ratio_reported(self):
        """Verify culling ratio is reasonable."""
        from gsplat.experimental.render.functional.gaussian_inference import (
            _cull_gaussians_higs,
        )
        N = 200
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        viewmat, K, w, h = _make_camera(device)
        _, _, ratio = _cull_gaussians_higs(
            means, quats, scales, opacities, colors,
            viewmat, K, w, h, None,
        )
        assert ratio > 0.1
        assert ratio < 1.0


class TestHigsAutogradFunction:
    """Direct tests for the _HigsAutogradFunction autograd Function."""

    @_skip_no_cuda
    def test_autograd_function_importable(self):
        """_HigsAutogradFunction can be imported and is callable."""
        from gsplat.experimental.render.functional.gaussian_inference import (
            _HigsAutogradFunction,
        )
        assert callable(_HigsAutogradFunction)
        assert _HigsAutogradFunction is not None

    @_skip_no_cuda
    def test_autograd_function_forward_shapes(self):
        """Forward returns correctly shaped outputs."""
        from gsplat.experimental.render.functional.gaussian_inference import (
            _HigsAutogradFunction,
        )
        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        viewmat, K, w, h = _make_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)

        frame, alpha = _HigsAutogradFunction.apply(
            means, quats, scales, opacities, colors,
            viewmats, Ks, w, h, None, 16,
            0.01, 1e10, 0.0, 0.3, None,
            "RGB", "pinhole",
            True, False,
        )
        assert frame.dim() == 4
        assert frame.shape == (1, h, w, 3)
        assert alpha.dim() == 4
        assert alpha.shape == (1, h, w, 1)
        assert frame.isfinite().all()
        assert alpha.isfinite().all()

    @_skip_no_cuda
    def test_autograd_function_backward_all_params(self):
        """Backward computes gradients for all 5 master parameters."""
        from gsplat.experimental.render.functional.gaussian_inference import (
            _HigsAutogradFunction,
        )
        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        viewmat, K, w, h = _make_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)

        frame, alpha = _HigsAutogradFunction.apply(
            means, quats, scales, opacities, colors,
            viewmats, Ks, w, h, None, 16,
            0.01, 1e10, 0.0, 0.3, None,
            "RGB", "pinhole",
            True, False,
        )
        frame.sum().backward()

        for name, t in [("means", means), ("quats", quats), ("scales", scales),
                         ("opacities", opacities), ("colors", colors)]:
            assert t.grad is not None, f"{name}.grad is None"
            assert t.grad.isfinite().all(), f"{name}.grad has non-finite values"
            assert t.grad.abs().sum() > 0, f"{name}.grad is all zeros"

    @_skip_no_cuda
    def test_autograd_function_no_culling(self):
        """Works without culling."""
        from gsplat.experimental.render.functional.gaussian_inference import (
            _HigsAutogradFunction,
        )
        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        viewmat, K, w, h = _make_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)

        frame, alpha = _HigsAutogradFunction.apply(
            means, quats, scales, opacities, colors,
            viewmats, Ks, w, h, None, 16,
            0.01, 1e10, 0.0, 0.3, None,
            "RGB", "pinhole",
            False, False,
        )
        frame.sum().backward()

        for name, t in [("means", means), ("quats", quats), ("scales", scales),
                         ("opacities", opacities), ("colors", colors)]:
            assert t.grad is not None
            assert t.grad.isfinite().all()
            assert t.grad.abs().sum() > 0

    @_skip_no_cuda
    def test_autograd_function_forward_aligned_with_frozen(self):
        """_HigsAutogradFunction forward matches frozen API output."""
        from gsplat.experimental import rasterize_gaussian_higs_frozen
        from gsplat.experimental.render.functional.gaussian_inference import (
            _HigsAutogradFunction,
        )
        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        viewmat, K, w, h = _make_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)

        frame_fn, alpha_fn = _HigsAutogradFunction.apply(
            means, quats, scales, opacities, colors,
            viewmats, Ks, w, h, None, 16,
            0.01, 1e10, 0.0, 0.3, None,
            "RGB", "pinhole",
            True, False,
        )

        result = rasterize_gaussian_higs_frozen(
            means, quats, scales, opacities, colors,
            viewmats=viewmats, Ks=Ks, width=w, height=h,
        )

        torch.testing.assert_close(frame_fn, result["frame"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(alpha_fn, result["alpha"], atol=1e-5, rtol=1e-5)


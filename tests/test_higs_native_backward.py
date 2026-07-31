"""Tests for the HiGS native CUDA backward (frozen topology, Stage B2).

Covers:
- forward RGB/alpha parity vs the standard gsplat fused rasterization;
- finite-difference gradients for means/quats/scales/opacities/RGB;
- torch.autograd.gradcheck on a small fully-visible scene;
- background (non-None) forward/backward;
- single- and multi-camera batches (no viewmats[0,0] hardcoding);
- empty / all / partial visible sets + zero gradient for invisible Gaussians;
- mixed precision (FP32 master -> FP16 packed buffers);
- SH degrees 0..3 and pre-activated RGB;
- SH compression rejection in the training path;
- native vs gsplat_recompute gradient comparison;
- explicit fallback when the CUDA extension is unavailable;
- pending-backward topology mutation raises;
- densify/prune optimizer state sync.
"""

import math

import pytest
import torch

device = torch.device("cuda:0")
_skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No CUDA device"
)


def _make_gaussians(N, d, device, seed=42, z=2.0):
    torch.manual_seed(seed)
    means = torch.randn(N, 3, device=device)
    means[:, 2] = means[:, 2].abs() + z
    quats = torch.randn(N, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.rand(N, 3, device=device) * 0.1 + 0.01
    opacities = torch.rand(N, device=device) * 0.8 + 0.1
    colors = torch.sigmoid(torch.randn(N, d, device=device))
    return means, quats, scales, opacities, colors


def _make_smooth_scene(N, device, seed=7):
    """Deterministic scene with Gaussians centered near pixel centers.

    Keeps projected footprints small (z ~ 9, sigma ~ 1.3 px) so tiny
    parameter perturbations never flip the forward's discrete tile
    binning, making finite-difference and gradcheck comparisons valid.
    """
    torch.manual_seed(seed)
    cols = 8
    fx = 64.0
    i = torch.arange(N, device=device) % cols
    j = torch.arange(N, device=device) // cols
    xs = 35.5 + 2.0 * i.float()
    ys = 19.5 + 2.0 * j.float()
    zs = 9.0 + 0.01 * torch.arange(N, device=device).float()
    means = torch.zeros(N, 3, device=device)
    means[:, 0] = (xs - 32.0) * zs / fx
    means[:, 1] = (ys - 24.0) * zs / fx
    means[:, 2] = zs
    quats = torch.zeros(N, 4, device=device)
    quats[:, 3] = 1.0
    sigmas = torch.tensor([0.19, 0.17, 0.15], device=device)
    scales = (sigmas[None, :] * zs[:, None] / fx).clone()
    opacities = torch.rand(N, device=device) * 0.6 + 0.25
    colors = torch.sigmoid(torch.randn(N, 3, device=device))
    return means, quats, scales, opacities, colors


def _make_camera(device, width=64, height=48, tx=0.0, ty=0.0):
    K = torch.tensor(
        [[64.0, 0.0, width / 2.0], [0.0, 64.0, height / 2.0], [0.0, 0.0, 1.0]],
        device=device,
    )
    viewmat = torch.eye(4, device=device)
    viewmat[0, 3] = tx
    viewmat[1, 3] = ty
    return viewmat, K, width, height


def _full_view_camera(device, width=64, height=48):
    """Camera looking straight down the -Z axis; Gaussians near origin visible."""
    K = torch.tensor(
        [[64.0, 0.0, width / 2.0], [0.0, 64.0, height / 2.0], [0.0, 0.0, 1.0]],
        device=device,
    )
    viewmat = torch.eye(4, device=device)
    return viewmat, K, width, height


def _require_ext():
    from gsplat.experimental.render.kernels import _backend

    if _backend._C is None or not hasattr(_backend._C, "higs_rasterize_backward"):
        pytest.skip("HiGS experimental CUDA extension not built")
    return _backend._C


def _run_frozen(params, viewmats, Ks, w, h, **kw):
    from gsplat.experimental import rasterize_gaussian_higs_frozen

    means, quats, scales, opacities, colors = params
    return rasterize_gaussian_higs_frozen(
        means, quats, scales, opacities, colors,
        viewmats=viewmats, Ks=Ks, width=w, height=h,
        freeze_topology=False, **kw,
    )


class TestForwardParity:
    @_skip_no_cuda
    def test_rgb_forward_matches_standard_gsplat(self):
        _require_ext()
        from gsplat.rendering import rasterization

        N = 64
        means, quats, scales, opacities, colors = _make_gaussians(N, 3, device)
        viewmat, K, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)

        res = _run_frozen(
            (means, quats, scales, opacities, colors),
            viewmats, Ks, w, h, backward_mode="higs_native",
        )
        ref, ref_alpha, _ = rasterization(
            means=means.unsqueeze(0), quats=quats.unsqueeze(0),
            scales=scales.unsqueeze(0), opacities=opacities.unsqueeze(0),
            colors=colors.unsqueeze(0),
            viewmats=viewmats, Ks=Ks, width=w, height=h,
            packed=True,
        )
        torch.testing.assert_close(res["frame"][0], ref[0][0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(res["alpha"][0], ref_alpha[0][0], atol=1e-4, rtol=1e-4)

    @_skip_no_cuda
    def test_rgb_forward_matches_with_background(self):
        _require_ext()
        from gsplat.rendering import rasterization

        N = 64
        means, quats, scales, opacities, colors = _make_gaussians(N, 3, device)
        viewmat, K, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)
        bg = torch.tensor([0.1, 0.2, 0.3], device=device)

        res = _run_frozen(
            (means, quats, scales, opacities, colors),
            viewmats, Ks, w, h, background=bg, backward_mode="higs_native",
        )
        ref, ref_alpha, _ = rasterization(
            means=means.unsqueeze(0), quats=quats.unsqueeze(0),
            scales=scales.unsqueeze(0), opacities=opacities.unsqueeze(0),
            colors=colors.unsqueeze(0),
            viewmats=viewmats, Ks=Ks, width=w, height=h,
            backgrounds=bg.unsqueeze(0).unsqueeze(0), packed=True,
        )
        torch.testing.assert_close(res["frame"][0], ref[0][0], atol=1e-4, rtol=1e-4)

    @_skip_no_cuda
    @pytest.mark.parametrize("sh_degree", [0, 1, 2, 3])
    def test_sh_forward_matches_standard_gsplat(self, sh_degree):
        _require_ext()
        from gsplat.rendering import rasterization

        N = 32
        K = (sh_degree + 1) ** 2
        means, quats, scales, opacities, _ = _make_gaussians(N, 3, device)
        torch.manual_seed(7)
        sh = (torch.randn(N, K, 3, device=device) * 0.3).contiguous()
        viewmat, Kc, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = Kc.unsqueeze(0).unsqueeze(0)

        res = _run_frozen(
            (means, quats, scales, opacities, sh),
            viewmats, Ks, w, h,
            sh_degree=sh_degree, backward_mode="higs_native",
        )
        ref, ref_alpha, _ = rasterization(
            means=means.unsqueeze(0), quats=quats.unsqueeze(0),
            scales=scales.unsqueeze(0), opacities=opacities.unsqueeze(0),
            colors=sh,
            viewmats=viewmats, Ks=Ks, width=w, height=h,
            sh_degree=sh_degree, packed=True,
        )
        torch.testing.assert_close(res["frame"][0], ref[0][0], atol=1e-4, rtol=1e-4)


class TestNativeBackwardGradients:
    @_skip_no_cuda
    def test_all_params_gradients_nonzero_finite(self):
        _require_ext()
        N = 64
        means, quats, scales, opacities, colors = _make_gaussians(N, 3, device)
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        viewmat, K, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)

        res = _run_frozen(
            (means, quats, scales, opacities, colors),
            viewmats, Ks, w, h, backward_mode="higs_native",
        )
        loss = (res["frame"] * torch.tensor([1.0, 2.0, 3.0], device=device)).sum()
        loss.backward()
        for name, t in [("means", means), ("quats", quats), ("scales", scales),
                        ("opacities", opacities), ("colors", colors)]:
            assert t.grad is not None, f"{name}.grad is None"
            assert t.grad.isfinite().all(), f"{name}.grad non-finite"
            assert t.grad.abs().sum() > 0, f"{name}.grad all zeros"

    @_skip_no_cuda
    def test_invisible_gaussians_get_zero_grad(self):
        _require_ext()
        N = 64
        means, quats, scales, opacities, colors = _make_gaussians(N, 3, device)
        # Move the last 20 Gaussians far behind the camera -> invisible.
        with torch.no_grad():
            means[N - 20 :, 2] = -50.0
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        viewmat, K, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)

        res = _run_frozen(
            (means, quats, scales, opacities, colors),
            viewmats, Ks, w, h, backward_mode="higs_native",
        )
        res["frame"].sum().backward()
        assert means.grad[N - 20 :].abs().sum() == 0
        assert quats.grad[N - 20 :].abs().sum() == 0
        assert scales.grad[N - 20 :].abs().sum() == 0
        assert opacities.grad[N - 20 :].abs().sum() == 0
        assert colors.grad[N - 20 :].abs().sum() == 0

    @_skip_no_cuda
    def test_finite_difference_means_quats_scales(self):
        _require_ext()
        N = 32
        means, quats, scales, opacities, colors = _make_smooth_scene(N, device, seed=7)
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        viewmat, K, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)

        def loss_fn():
            res = _run_frozen(
                (means, quats, scales, opacities, colors),
                viewmats, Ks, w, h,
                backward_mode="higs_native", enable_culling=False,
            )
            return res["frame"].sum()

        loss = loss_fn()
        loss.backward()

        eps = 1e-3
        tol = 2e-2

        def fd_check(param, grad, idx):
            base = loss_fn().item()
            flat = param.detach().clone().flatten()
            flat[idx] += eps
            with torch.no_grad():
                param.copy_(flat.view_as(param))
            up = loss_fn().item()
            flat[idx] -= 2 * eps
            with torch.no_grad():
                param.copy_(flat.view_as(param))
            down = loss_fn().item()
            flat[idx] += eps
            with torch.no_grad():
                param.copy_(flat.view_as(param))
            numeric = (up - down) / (2 * eps)
            analytic = grad.flatten()[idx].item()
            assert abs(numeric - analytic) <= tol * max(1.0, abs(numeric)), (
                f"idx {idx}: numeric={numeric} analytic={analytic}"
            )

        for name, t, g in [("means", means, means.grad), ("quats", quats, quats.grad),
                           ("scales", scales, scales.grad), ("opacities", opacities, opacities.grad),
                           ("colors", colors, colors.grad)]:
            for idx in [0, 1, t.numel() // 2, t.numel() - 1]:
                if g.flatten()[idx].abs().item() < 1e-8:
                    continue
                fd_check(t, g, idx)

    @_skip_no_cuda
    def test_gradcheck_small_fully_visible(self):
        _require_ext()
        N = 8
        means, quats, scales, opacities, colors = _make_smooth_scene(N, device, seed=11)
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        viewmat, K, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)
        # Small scene fully in view, no culling: the function is smooth.
        torch.autograd.gradcheck(
            lambda *xs: _run_frozen(xs, viewmats, Ks, w, h,
                                    backward_mode="higs_native",
                                    enable_culling=False)["frame"],
            (means, quats, scales, opacities, colors),
            eps=1e-3, atol=1e-2, rtol=1e-2, fast_mode=False,
        )

    @_skip_no_cuda
    def test_background_gradient(self):
        _require_ext()
        N = 32
        means, quats, scales, opacities, colors = _make_gaussians(N, 3, device)
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        bg = torch.tensor([0.1, 0.5, 0.9], device=device, requires_grad=True)
        viewmat, K, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)

        res = _run_frozen(
            (means, quats, scales, opacities, colors),
            viewmats, Ks, w, h, background=bg, backward_mode="higs_native",
        )
        res["frame"].sum().backward()
        assert bg.grad is not None
        assert bg.grad.isfinite().all()
        assert (bg.grad > 0).all()  # sum loss -> positive bg grad

    @_skip_no_cuda
    def test_alpha_output_backward(self):
        _require_ext()
        N = 32
        means, quats, scales, opacities, colors = _make_gaussians(N, 3, device)
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        viewmat, K, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)

        res = _run_frozen(
            (means, quats, scales, opacities, colors),
            viewmats, Ks, w, h, backward_mode="higs_native",
        )
        res["alpha"].sum().backward()
        for name, t in [("means", means), ("quats", quats), ("scales", scales),
                        ("opacities", opacities), ("colors", colors)]:
            assert t.grad is not None
            assert t.grad.isfinite().all()
        assert opacities.grad.abs().sum() > 0

    @_skip_no_cuda
    def test_multi_camera_batch(self):
        _require_ext()
        N = 48
        means, quats, scales, opacities, colors = _make_gaussians(N, 3, device)
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        v1, K, w, h = _full_view_camera(device)
        v2 = v1.clone()
        v2[0, 3] = 0.5
        v3 = v1.clone()
        v3[1, 3] = -0.5
        viewmats = torch.stack([v1, v2, v3]).unsqueeze(0)  # [1, 3, 4, 4]
        Ks = torch.stack([K, K, K]).unsqueeze(0)

        res = _run_frozen(
            (means, quats, scales, opacities, colors),
            viewmats, Ks, w, h, backward_mode="higs_native",
        )
        assert res["frame"].shape == (1, 3, h, w, 3)
        res["frame"].sum().backward()
        for name, t in [("means", means), ("quats", quats), ("scales", scales),
                        ("opacities", opacities), ("colors", colors)]:
            assert t.grad is not None and t.grad.isfinite().all()

    @_skip_no_cuda
    def test_empty_visible_set_pure_background(self):
        _require_ext()
        N = 16
        means, quats, scales, opacities, colors = _make_gaussians(N, 3, device)
        # All Gaussians far behind the camera -> empty visible set.
        with torch.no_grad():
            means[:, 2] = -50.0
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        bg = torch.tensor([0.25, 0.5, 0.75], device=device, requires_grad=True)
        viewmat, K, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)

        res = _run_frozen(
            (means, quats, scales, opacities, colors),
            viewmats, Ks, w, h, background=bg, backward_mode="higs_native",
        )
        assert res["frame"].shape == (1, h, w, 3)
        torch.testing.assert_close(
            res["frame"][0], bg.view(1, 1, 3).expand(h, w, 3), atol=1e-6, rtol=1e-6
        )
        res["frame"].sum().backward()
        for t in [means, quats, scales, opacities, colors]:
            assert t.grad is not None and t.grad.abs().sum() == 0
        assert bg.grad is not None and bg.grad.abs().sum() > 0

    @_skip_no_cuda
    def test_partial_visibility_ratio_metadata(self):
        _require_ext()
        N = 200
        means, quats, scales, opacities, colors = _make_gaussians(N, 3, device)
        viewmat, K, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)

        res = _run_frozen(
            (means, quats, scales, opacities, colors),
            viewmats, Ks, w, h, backward_mode="higs_native",
        )
        md = res["metadata"]
        assert 0.0 < md["culling_ratio"] < 1.0
        assert md["n_visible"] == int(round((1.0 - md["culling_ratio"]) * N))
        assert md["backward_backend"] == "higs_native"
        assert md["n_gaussians"] == N

    @_skip_no_cuda
    def test_mixed_precision_packed_dtype_metadata(self):
        _require_ext()
        from gsplat.experimental.render.functional.gaussian_inference import (
            create_higs_renderer,
        )

        N = 32
        means, quats, scales, opacities, colors = _make_gaussians(N, 3, device)
        viewmat, K, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)
        handle = create_higs_renderer(means, quats, scales, opacities, colors)
        try:
            assert handle.qso_packed.dtype == torch.float16
            res = _run_frozen(
                (means, quats, scales, opacities, colors),
                viewmats, Ks, w, h,
                backward_mode="higs_native",
                use_higs_culling=True, scene=handle,
            )
            assert res["metadata"]["packed_dtype"] == "torch.float16"
            assert res["metadata"]["scene_version"] == handle.version
        finally:
            handle.release()


class TestSHAndCompression:
    @_skip_no_cuda
    @pytest.mark.parametrize("sh_degree", [0, 1, 2, 3])
    def test_sh_gradients_nonzero(self, sh_degree):
        _require_ext()
        N = 32
        K = (sh_degree + 1) ** 2
        means, quats, scales, opacities, _ = _make_gaussians(N, 3, device)
        torch.manual_seed(11)
        sh = (torch.randn(N, K, 3, device=device) * 0.3).contiguous()
        for t in [means, quats, scales, opacities, sh]:
            t.requires_grad_(True)
        viewmat, Kc, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = Kc.unsqueeze(0).unsqueeze(0)

        res = _run_frozen(
            (means, quats, scales, opacities, sh),
            viewmats, Ks, w, h,
            sh_degree=sh_degree, backward_mode="higs_native",
        )
        res["frame"].sum().backward()
        for name, t in [("means", means), ("quats", quats), ("scales", scales),
                        ("opacities", opacities), ("sh", sh)]:
            assert t.grad is not None, f"{name}.grad is None"
            assert t.grad.isfinite().all(), f"{name}.grad non-finite"
            assert t.grad.abs().sum() > 0, f"{name}.grad all zeros"

    @_skip_no_cuda
    def test_sh_native_vs_recompute_gradients_agree(self):
        """Full SH chain (degree 3) incl. the clamp_min activation; forces the
        clamp active with large negative coefficients so the backward mask on
        forward colors_eval is exercised."""
        _require_ext()
        N = 48
        K = 16
        means, quats, scales, opacities, _ = _make_gaussians(N, 3, device)
        torch.manual_seed(21)
        sh = (torch.randn(N, K, 3, device=device) * 0.3).contiguous()
        # Force several Gaussians into the clamped region (sph + 0.5 < 0).
        with torch.no_grad():
            sh[:6, 0, 0] = -3.0
            sh[6:12, 3, 1] = -3.0
            sh[12:16, 8, 2] = -3.0
        for t in [means, quats, scales, opacities, sh]:
            t.requires_grad_(True)
        viewmat, Kc, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = Kc.unsqueeze(0).unsqueeze(0)
        target = torch.rand(1, h, w, 3, device=device) * 0.5

        def grads(mode):
            for t in [means, quats, scales, opacities, sh]:
                t.grad = None
            res = _run_frozen(
                (means, quats, scales, opacities, sh),
                viewmats, Ks, w, h, sh_degree=3, backward_mode=mode,
            )
            loss = (res["frame"] - target).pow(2).mean()
            loss.backward()
            return [t.grad.clone() for t in [means, quats, scales, opacities, sh]]

        g_native = grads("higs_native")
        g_recomp = grads("gsplat_recompute")
        for name, a, b in zip(
            ["means", "quats", "scales", "opacities", "sh"], g_native, g_recomp
        ):
            torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)

    @_skip_no_cuda
    def test_sh_finite_difference(self):
        """FD check on SH coefficients (degree 3), including a clamped row."""
        _require_ext()
        N = 16
        K = 16
        means, quats, scales, opacities, _ = _make_gaussians(N, 3, device)
        torch.manual_seed(22)
        sh = (torch.randn(N, K, 3, device=device) * 0.3).contiguous()
        with torch.no_grad():
            sh[0, 0, 0] = -3.0  # clamped channel
        for t in [means, quats, scales, opacities, sh]:
            t.requires_grad_(True)
        viewmat, Kc, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = Kc.unsqueeze(0).unsqueeze(0)

        def loss_fn():
            res = _run_frozen(
                (means, quats, scales, opacities, sh),
                viewmats, Ks, w, h, sh_degree=3, backward_mode="higs_native",
            )
            return (res["frame"] * torch.tensor([1.0, 2.0, 3.0], device=device)).sum()

        loss_fn().backward()
        eps, tol = 1e-3, 3e-2
        for idx in [0, 1, K * 3 // 2, K * 3 - 1, sh[0].numel() // 2]:
            g = sh.grad.flatten()[idx].item()
            if abs(g) < 1e-8:
                continue
            base = loss_fn().item()
            flat = sh.detach().clone().flatten()
            flat[idx] += eps
            with torch.no_grad():
                sh.copy_(flat.view_as(sh))
            up = loss_fn().item()
            flat[idx] -= 2 * eps
            with torch.no_grad():
                sh.copy_(flat.view_as(sh))
            down = loss_fn().item()
            flat[idx] += eps
            with torch.no_grad():
                sh.copy_(flat.view_as(sh))
            numeric = (up - down) / (2 * eps)
            assert abs(numeric - g) <= tol * max(1.0, abs(numeric)), (
                f"idx {idx}: numeric={numeric} analytic={g}"
            )

    @_skip_no_cuda
    def test_sh_compression_rejected_in_training(self):
        _require_ext()
        N = 16
        means, quats, scales, opacities, colors = _make_gaussians(N, 3, device)
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        viewmat, K, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)

        with pytest.raises(ValueError, match="SH compression"):
            _run_frozen(
                (means, quats, scales, opacities, colors),
                viewmats, Ks, w, h,
                backward_mode="higs_native",
                sh_compression_mode="packed_32b",
            )


class TestNativeVsRecompute:
    @_skip_no_cuda
    def test_native_vs_recompute_gradients_agree(self):
        _require_ext()
        N = 48
        means, quats, scales, opacities, colors = _make_gaussians(N, 3, device)
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        viewmat, K, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)
        target = torch.rand(1, h, w, 3, device=device) * 0.5

        def grads(mode):
            for t in [means, quats, scales, opacities, colors]:
                t.grad = None
            res = _run_frozen(
                (means, quats, scales, opacities, colors),
                viewmats, Ks, w, h, backward_mode=mode,
            )
            loss = (res["frame"] - target).pow(2).mean()
            loss.backward()
            return [t.grad.clone() for t in [means, quats, scales, opacities, colors]]

        g_native = grads("higs_native")
        g_recomp = grads("gsplat_recompute")
        for name, a, b in zip(
            ["means", "quats", "scales", "opacities", "colors"], g_native, g_recomp
        ):
            torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)


class TestFallbackAndErrors:
    @_skip_no_cuda
    def test_fallback_when_extension_unavailable(self, monkeypatch):
        import gsplat.experimental.render.functional.gaussian_inference as m

        monkeypatch.setattr(m, "_higs_backend_available", lambda: False)
        N = 32
        means, quats, scales, opacities, colors = _make_gaussians(N, 3, device)
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        viewmat, K, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)

        res = _run_frozen(
            (means, quats, scales, opacities, colors),
            viewmats, Ks, w, h, backward_mode="higs_native",
        )
        assert res["metadata"]["backward_backend"] == "gsplat_recompute"
        res["frame"].sum().backward()
        for name, t in [("means", means), ("quats", quats), ("scales", scales),
                        ("opacities", opacities), ("colors", colors)]:
            assert t.grad is not None and t.grad.isfinite().all()

    @_skip_no_cuda
    def test_recompute_mode_metadata(self):
        _require_ext()
        N = 16
        means, quats, scales, opacities, colors = _make_gaussians(N, 3, device)
        viewmat, K, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)

        res = _run_frozen(
            (means, quats, scales, opacities, colors),
            viewmats, Ks, w, h, backward_mode="gsplat_recompute",
        )
        assert res["metadata"]["backward_backend"] == "gsplat_recompute"

    @_skip_no_cuda
    def test_invalid_backward_mode_raises(self):
        _require_ext()
        N = 16
        means, quats, scales, opacities, colors = _make_gaussians(N, 3, device)
        viewmat, K, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)
        with pytest.raises(ValueError, match="backward_mode"):
            _run_frozen(
                (means, quats, scales, opacities, colors),
                viewmats, Ks, w, h, backward_mode="bogus",
            )

    @_skip_no_cuda
    def test_native_rejects_non_rgb(self):
        _require_ext()
        N = 16
        means, quats, scales, opacities, colors = _make_gaussians(N, 3, device)
        viewmat, K, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)
        with pytest.raises(ValueError, match="RGB"):
            _run_frozen(
                (means, quats, scales, opacities, colors),
                viewmats, Ks, w, h, backward_mode="higs_native",
                render_mode="RGB+D",
            )


class TestDynamicTopology:
    @_skip_no_cuda
    def test_pending_backward_mutation_raises(self):
        _require_ext()
        from gsplat.experimental import rasterize_gaussian_higs_dynamic
        from gsplat.experimental.render.functional.gaussian_inference import (
            _HIGS_DYNAMIC_SCENE,
            _densify_gaussians,
        )

        _HIGS_DYNAMIC_SCENE.reset()
        N = 24
        means, quats, scales, opacities, colors = _make_gaussians(N, 3, device)
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        viewmat, K, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)

        res = rasterize_gaussian_higs_dynamic(
            means, quats, scales, opacities, colors,
            viewmats=viewmats, Ks=Ks, width=w, height=h,
            backward_mode="higs_native",
        )
        # Backward NOT run yet -> topology mutation must raise.
        with pytest.raises(RuntimeError, match="pending"):
            _HIGS_DYNAMIC_SCENE.mark_dirty()

        res["frame"].sum().backward()
        # After backward, mutation is allowed.
        _HIGS_DYNAMIC_SCENE.mark_dirty()

    @_skip_no_cuda
    def test_densify_optimizer_state_sync(self):
        from gsplat.experimental.render.functional.gaussian_inference import (
            _densify_gaussians,
            sync_optimizer_state_for_topology_change,
        )

        N = 16
        means, quats, scales, opacities, colors = _make_gaussians(N, 3, device)
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        opt = torch.optim.Adam(
            [means, quats, scales, opacities, colors], lr=1e-2
        )
        # Run a fake step so Adam state exists.
        opt.zero_grad()
        means.grad = torch.randn_like(means) * 1e-3
        quats.grad = torch.randn_like(quats) * 1e-3
        scales.grad = torch.randn_like(scales) * 1e-3
        opacities.grad = torch.randn_like(opacities) * 1e-3
        colors.grad = torch.randn_like(colors) * 1e-3
        opt.step()

        grads = torch.randn(N, 3, device=device) * 0.001
        new_m, new_q, new_s, new_o, new_c = _densify_gaussians(
            means, quats, scales, opacities, colors, grads, threshold=0.0005
        )
        N_new = new_m.shape[0]
        mask = grads.norm(dim=-1) > 0.0005
        dup_idx = mask.nonzero().flatten()
        old_to_new = torch.cat(
            [torch.arange(N, device=device), dup_idx]
        ).to(torch.long)
        src_exp_avg = opt.state[means]["exp_avg"].clone()

        sync_optimizer_state_for_topology_change(
            opt, old_to_new,
            means=(means, new_m), quats=(quats, new_q),
            scales=(scales, new_s), opacities=(opacities, new_o),
            colors=(colors, new_c),
        )

        group = opt.param_groups[0]
        p_means = group["params"][0]
        assert p_means is new_m
        state = opt.state[p_means]
        assert "exp_avg" in state
        assert state["exp_avg"].shape == (N_new, 3)
        # duplicated rows are copies of their source rows
        src = src_exp_avg[dup_idx]
        dup = state["exp_avg"][N:]
        torch.testing.assert_close(dup, src, atol=1e-6, rtol=1e-6)

    @_skip_no_cuda
    def test_prune_optimizer_state_sync(self):
        from gsplat.experimental.render.functional.gaussian_inference import (
            _prune_gaussians,
            sync_optimizer_state_for_topology_change,
        )

        N = 16
        means, quats, scales, opacities, colors = _make_gaussians(N, 3, device)
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        opt = torch.optim.Adam([means], lr=1e-2)
        opt.zero_grad()
        means.grad = torch.randn_like(means) * 1e-3
        opt.step()

        with torch.no_grad():
            opacities[:5] = 0.001
        new_m, new_q, new_s, new_o, new_c = _prune_gaussians(
            means, quats, scales, opacities, colors, opacity_threshold=0.01
        )
        mask = (opacities > 0.01).to(torch.long)
        old_to_new = mask.nonzero().flatten()

        src_exp_avg = opt.state[means]["exp_avg"].clone()
        sync_optimizer_state_for_topology_change(
            opt, old_to_new, means=(means, new_m),
        )
        group = opt.param_groups[0]
        p_means = group["params"][0]
        assert p_means is new_m
        state = opt.state[p_means]
        assert state["exp_avg"].shape == (new_m.shape[0], 3)
        torch.testing.assert_close(
            state["exp_avg"], src_exp_avg[old_to_new], atol=1e-6, rtol=1e-6
        )


class TestStaticApiSurface:
    """Static/API-surface tests that run without a CUDA device.

    These intentionally do NOT use ``_skip_no_cuda``: the completion criteria
    require static/API tests to remain runnable in no-CUDA environments while
    only CUDA-specific tests skip cleanly.
    """

    def test_stage_apis_importable(self):
        from gsplat.experimental import (
            rasterize_gaussian_higs_dynamic,
            rasterize_gaussian_higs_frozen,
            rasterize_gaussian_higs_trainable,
        )

        assert callable(rasterize_gaussian_higs_trainable)
        assert callable(rasterize_gaussian_higs_frozen)
        assert callable(rasterize_gaussian_higs_dynamic)

    def test_backward_mode_default_native(self):
        import inspect

        from gsplat.experimental import (
            rasterize_gaussian_higs_dynamic,
            rasterize_gaussian_higs_frozen,
        )

        for fn in (rasterize_gaussian_higs_frozen, rasterize_gaussian_higs_dynamic):
            params = inspect.signature(fn).parameters
            assert params["backward_mode"].default == "higs_native"
            assert params["sh_compression_mode"].default == "none"

    def test_backend_probe_returns_bool(self):
        from gsplat.experimental.render.functional.gaussian_inference import (
            _higs_backend_available,
        )

        assert isinstance(_higs_backend_available(), bool)

    def test_metadata_keys_in_module_source(self):
        import inspect

        from gsplat.experimental.render.functional import gaussian_inference as m

        src = inspect.getsource(m)
        required = {
            "backward_backend",
            "scene_version",
            "n_gaussians",
            "n_visible",
            "culling_ratio",
            "topology_rebuilt",
            "packed_dtype",
        }
        missing = {k for k in required if '"%s"' % k not in src}
        assert not missing, f"metadata keys missing from module source: {missing}"

    def test_handle_and_scene_api_surface(self):
        from gsplat.experimental.render.functional.gaussian_inference import (
            _HigsDynamicScene,
            HigsRendererHandle,
        )

        for name in ("mark_dirty", "rebuild", "release", "version"):
            assert hasattr(HigsRendererHandle, name)
        for name in ("scene_version", "next_version", "validate_count"):
            assert hasattr(_HigsDynamicScene, name)


class TestCullingBoundaryFD:
    """FD verification at the discrete culling boundaries (goal item 5).

    Visibility is stop-gradient: culled Gaussians must receive exactly zero
    gradient on all parameters and the mask itself is not differentiated.
    These tests place Gaussians at the near/far planes, at the projection
    (image) boundary, and around the radius-clip threshold, then verify zero
    grads on the culled side and FD-consistent grads on the visible side.
    """

    @staticmethod
    def _place(device, img_x, img_y, z, fx=64.0, cx=32.0, cy=24.0):
        return torch.tensor(
            [[(img_x - cx) * z / fx, (img_y - cy) * z / fx, z]], device=device
        )

    @staticmethod
    def _fd_check(param, loss_fn, idx, eps=1e-3, tol=2e-2):
        base = loss_fn().item()
        flat = param.detach().clone().flatten()
        flat[idx] += eps
        with torch.no_grad():
            param.copy_(flat.view_as(param))
        up = loss_fn().item()
        flat[idx] -= 2 * eps
        with torch.no_grad():
            param.copy_(flat.view_as(param))
        down = loss_fn().item()
        flat[idx] += eps
        with torch.no_grad():
            param.copy_(flat.view_as(param))
        numeric = (up - down) / (2 * eps)
        analytic = param.grad.flatten()[idx].item()
        assert abs(numeric - analytic) <= tol * max(1.0, abs(numeric)), (
            f"idx {idx}: numeric={numeric} analytic={analytic}"
        )

    @staticmethod
    def _run(means, quats, scales, opacities, colors, viewmats, Ks, w, h, **kw):
        return _run_frozen(
            (means, quats, scales, opacities, colors),
            viewmats, Ks, w, h, backward_mode="higs_native", **kw,
        )

    @_skip_no_cuda
    def test_near_plane_culling_fd(self):
        _require_ext()
        N = 8
        means, quats, scales, opacities, colors = _make_smooth_scene(N, device, seed=5)
        near_plane = 0.5
        with torch.no_grad():
            extra_m = self._place(device, 32.0, 24.0, near_plane + 0.05)
            extra_q = torch.zeros(1, 4, device=device)
            extra_q[0, 3] = 1.0
            extra_s = torch.tensor([[0.19, 0.17, 0.15]], device=device) * (near_plane + 0.05) / 9.0
            means = torch.cat([means, extra_m])
            quats = torch.cat([quats, extra_q])
            scales = torch.cat([scales, extra_s])
            opacities = torch.cat([opacities, torch.tensor([0.8], device=device)])
            colors = torch.cat([colors, torch.tensor([[0.9, 0.1, 0.1]], device=device)])
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        viewmat, K, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)
        i = N  # boundary gaussian index

        def loss_fn():
            res = self._run(
                means, quats, scales, opacities, colors,
                viewmats, Ks, w, h, use_higs_culling=True, near_plane=near_plane,
            )
            return res["frame"].sum()

        # Visible side (z just above near plane): grad flows and matches FD.
        loss_fn().backward()
        assert means.grad[i].abs().sum() > 0
        assert scales.grad[i].abs().sum() > 0
        self._fd_check(means, loss_fn, i * 3 + 2)  # dL/dz
        self._fd_check(scales, loss_fn, i * 3, eps=1e-4)  # dL/dscale_x

        # Culled side (z behind the near plane): gradient is exactly zero.
        for t in [means, quats, scales, opacities, colors]:
            t.grad = None
        with torch.no_grad():
            means[i, 2] = near_plane - 0.05
        res = self._run(
            means, quats, scales, opacities, colors,
            viewmats, Ks, w, h, use_higs_culling=True, near_plane=near_plane,
        )
        res["frame"].sum().backward()
        assert res["metadata"]["culling_ratio"] > 0
        for t in [means, quats, scales, opacities, colors]:
            assert t.grad[i].abs().sum() == 0, f"{t.shape} boundary row not zero"
        assert means.grad[:i].abs().sum() > 0  # other Gaussians still visible

    @_skip_no_cuda
    def test_far_plane_culling_fd(self):
        _require_ext()
        N = 8
        means, quats, scales, opacities, colors = _make_smooth_scene(N, device, seed=6)
        far_plane = 50.0
        with torch.no_grad():
            extra_m = self._place(device, 32.0, 24.0, far_plane - 1.0)
            extra_q = torch.zeros(1, 4, device=device)
            extra_q[0, 3] = 1.0
            extra_s = torch.tensor([[0.19, 0.17, 0.15]], device=device) * (far_plane - 1.0) / 9.0
            means = torch.cat([means, extra_m])
            quats = torch.cat([quats, extra_q])
            scales = torch.cat([scales, extra_s])
            opacities = torch.cat([opacities, torch.tensor([0.8], device=device)])
            colors = torch.cat([colors, torch.tensor([[0.1, 0.9, 0.1]], device=device)])
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        viewmat, K, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)
        i = N

        def loss_fn():
            res = self._run(
                means, quats, scales, opacities, colors,
                viewmats, Ks, w, h, use_higs_culling=True, far_plane=far_plane,
            )
            return res["frame"].sum()

        loss_fn().backward()
        assert means.grad[i].abs().sum() > 0
        self._fd_check(means, loss_fn, i * 3 + 2)

        for t in [means, quats, scales, opacities, colors]:
            t.grad = None
        with torch.no_grad():
            means[i, 2] = far_plane + 2.0  # beyond far plane -> culled
        res = self._run(
            means, quats, scales, opacities, colors,
            viewmats, Ks, w, h, use_higs_culling=True, far_plane=far_plane,
        )
        res["frame"].sum().backward()
        assert res["metadata"]["culling_ratio"] > 0
        for t in [means, quats, scales, opacities, colors]:
            assert t.grad[i].abs().sum() == 0
        assert means.grad[:i].abs().sum() > 0

    @_skip_no_cuda
    def test_radius_clip_culling_fd(self):
        _require_ext()
        N = 8
        means, quats, scales, opacities, colors = _make_smooth_scene(N, device, seed=7)
        radius_clip = 2.0  # px (probed threshold)
        with torch.no_grad():
            # Big Gaussian: projected radius >> radius_clip -> always visible.
            big_m = self._place(device, 20.0, 24.0, 9.0)
            big_q = torch.zeros(1, 4, device=device)
            big_q[0, 3] = 1.0
            big_s = torch.tensor([[1.5, 1.4, 1.3]], device=device) * (9.0 / 64.0)
            # Small Gaussian: projected radius < radius_clip when clipping on.
            small_m = self._place(device, 44.0, 24.0, 9.0)
            small_s = torch.tensor([[0.05, 0.05, 0.05]], device=device) * (9.0 / 64.0)
            means = torch.cat([means, big_m, small_m])
            quats = torch.cat([quats, big_q, big_q])
            scales = torch.cat([scales, big_s, small_s])
            opacities = torch.cat([opacities, torch.tensor([0.8, 0.8], device=device)])
            colors = torch.cat([colors, torch.tensor([[0.9, 0.9, 0.1], [0.1, 0.1, 0.9]], device=device)])
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        viewmat, K, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)
        ibig, ismall = N, N + 1

        # radius_clip=0: both visible.
        res0 = self._run(
            means, quats, scales, opacities, colors,
            viewmats, Ks, w, h, use_higs_culling=True, radius_clip=0.0,
        )
        res0["frame"].sum().backward()
        assert means.grad[ibig].abs().sum() > 0
        assert means.grad[ismall].abs().sum() > 0

        # radius_clip=2.0: the small Gaussian is culled, the big one stays visible.
        def loss_fn():
            res = self._run(
                means, quats, scales, opacities, colors,
                viewmats, Ks, w, h, use_higs_culling=True, radius_clip=radius_clip,
            )
            return res["frame"].sum()

        for t in [means, quats, scales, opacities, colors]:
            t.grad = None
        res = loss_fn()
        res.backward()
        md = self._run(
            means, quats, scales, opacities, colors,
            viewmats, Ks, w, h, use_higs_culling=True, radius_clip=radius_clip,
        )["metadata"]
        assert md["culling_ratio"] > 0
        assert means.grad[ismall].abs().sum() == 0
        assert means.grad[ibig].abs().sum() > 0
        self._fd_check(scales, loss_fn, ibig * 3)  # dL/dscale_x of the big one

    @_skip_no_cuda
    def test_projection_boundary_culling_fd(self):
        _require_ext()
        N = 8
        means, quats, scales, opacities, colors = _make_smooth_scene(N, device, seed=8)
        with torch.no_grad():
            # Off-screen Gaussian (image x = -6 -> fully outside the frame).
            off_m = self._place(device, -6.0, 24.0, 9.0)
            # On-screen boundary Gaussian (image x = 2 px).
            on_m = self._place(device, 2.0, 24.0, 9.0)
            q1 = torch.zeros(1, 4, device=device)
            q1[0, 3] = 1.0
            s1 = torch.tensor([[0.19, 0.17, 0.15]], device=device) * (9.0 / 64.0)
            means = torch.cat([means, off_m, on_m])
            quats = torch.cat([quats, q1, q1])
            scales = torch.cat([scales, s1, s1])
            opacities = torch.cat([opacities, torch.tensor([0.8, 0.8], device=device)])
            colors = torch.cat([colors, torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9]], device=device)])
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        viewmat, K, w, h = _full_view_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)
        ioff, ion = N, N + 1

        def loss_fn():
            res = self._run(
                means, quats, scales, opacities, colors,
                viewmats, Ks, w, h, use_higs_culling=True,
            )
            return res["frame"].sum()

        # Off-screen -> culled with zero grad; on-screen -> visible with FD match.
        res = self._run(
            means, quats, scales, opacities, colors,
            viewmats, Ks, w, h, use_higs_culling=True,
        )
        res["frame"].sum().backward()
        assert res["metadata"]["culling_ratio"] > 0
        assert means.grad[ioff].abs().sum() == 0
        assert means.grad[ion].abs().sum() > 0
        self._fd_check(means, loss_fn, ion * 3)  # dL/dx of the on-screen one

        # Moving the off-screen Gaussian across the projection boundary makes
        # it visible again (discrete flip; the mask itself is stop-gradient).
        for t in [means, quats, scales, opacities, colors]:
            t.grad = None
        with torch.no_grad():
            means[ioff, 0] = (2.0 - 32.0) * 9.0 / 64.0  # image x = 2 px
        res2 = self._run(
            means, quats, scales, opacities, colors,
            viewmats, Ks, w, h, use_higs_culling=True,
        )
        res2["frame"].sum().backward()
        assert means.grad[ioff].abs().sum() > 0


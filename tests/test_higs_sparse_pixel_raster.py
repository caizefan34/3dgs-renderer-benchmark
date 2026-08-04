"""Tests for the renderer-level sparse-pixel rasterization backend.

``higs_sparse_px`` implements Speedy-Splat-style pixel-sparse rasterization
with upstream gsplat's sparse kernels (``build_sparse_tile_layout`` +
``isect_tiles_sparse`` + ``rasterize_to_pixels_sparse``): projection + SH run
for every camera, but only the pixels drawn by the per-step iid Bernoulli
mask are intersected/rasterized. The packed output is compared against the
reference gathered at the same pixel indices (R57: renderer-level
finer-than-tile sampling, the remaining M6 quality lever after R53/R54
closed the loss-signal side).
"""

import importlib.util
from pathlib import Path

import pytest
import torch

_BENCH = Path(__file__).resolve().parents[1] / "benchmark" / "run_higs_train_benchmark.py"
_spec = importlib.util.spec_from_file_location("run_higs_train_benchmark", _BENCH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

from higs_skip_helpers import (  # noqa: E402
    skipif_higs_module_unavailable as _skip_no_module,
    skipif_higs_unavailable as _skip_no_cuda,
)


# --------------------------------------------------------------------------
# CPU-safe unit tests
# --------------------------------------------------------------------------

class TestPackedPixelL1:
    def test_full_grid_equals_full_l1(self):
        torch.manual_seed(0)
        C, H, W = 2, 16, 16
        ref = torch.rand(C, H, W, 3)
        flat = torch.arange(C * H * W)
        frame = ref.reshape(-1, 3)[flat]  # packed "render" = exact ref
        est = _mod._packed_pixel_l1_loss(frame, ref, None, flat)
        assert torch.allclose(est, torch.tensor(0.0), atol=1e-6)

    def test_unbiased_estimate_at_ratio_0_35(self):
        torch.manual_seed(1)
        C, H, W = 1, 64, 64
        frame = torch.rand(1, H, W, 3)
        ref = torch.rand(1, H, W, 3)
        full = _mod._l1_loss(frame, ref).item()
        draws = []
        for _ in range(50):
            sel = torch.rand(C, H, W) < 0.35
            ids, rr, cc = sel.nonzero(as_tuple=True)
            flat = ids * H * W + rr * W + cc
            est = _mod._packed_pixel_l1_loss(
                frame.reshape(-1, 3)[flat], ref, ids, flat
            ).item()
            draws.append(est)
        assert abs(sum(draws) / len(draws) - full) < 0.01

    def test_pixel_order_independent_of_ref_gather(self):
        # The packed loss gathers ref at the same flat indices the renderer
        # used, so a scrambled pixel list must still recover the right values.
        torch.manual_seed(2)
        H, W = 8, 8
        ref = torch.rand(H, W, 3)
        flat = torch.arange(H * W)
        perm = torch.randperm(H * W)
        packed = ref.reshape(-1, 3)[perm]  # renderer packed in perm order
        est = _mod._packed_pixel_l1_loss(packed, ref, None, perm)
        assert torch.allclose(est, torch.tensor(0.0), atol=1e-6)


class TestPixelRasterRatioFlag:
    def test_parser_default_0_35(self):
        parser = _mod.build_arg_parser()
        args = parser.parse_args([
            "--base-dir", "datasets/processed",
            "--scene", "mipnerf360/garden",
        ])
        assert args.pixel_raster_ratio == 0.35

    def test_parser_accepts_value(self):
        parser = _mod.build_arg_parser()
        args = parser.parse_args([
            "--base-dir", "datasets/processed",
            "--scene", "mipnerf360/garden",
            "--pixel-raster-ratio", "0.5",
        ])
        assert args.pixel_raster_ratio == 0.5

    def test_run_backend_accepts_kwarg(self):
        import inspect
        sig = inspect.signature(_mod.run_backend)
        assert "pixel_raster_ratio" in sig.parameters
        assert sig.parameters["pixel_raster_ratio"].default == 0.35


# --------------------------------------------------------------------------
# CUDA-gated integration tests
# --------------------------------------------------------------------------

def _make_gaussians(N, device, seed=42):
    torch.manual_seed(seed)
    means = torch.randn(N, 3, device=device)
    means[:, 2] = means[:, 2].abs() + 3.0
    quats = torch.randn(N, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.rand(N, 3, device=device) * 0.2 + 0.05
    opacities = torch.rand(N, device=device) * 0.8 + 0.1
    sh = torch.randn(N, 16, 3, device=device) * 0.05
    return means, quats, scales, opacities, sh


def _make_cameras(device, C=2, width=64, height=48):
    K = torch.tensor(
        [[64.0, 0.0, width / 2.0], [0.0, 64.0, height / 2.0], [0.0, 0.0, 1.0]],
        device=device,
    )
    viewmats = torch.eye(4, device=device).unsqueeze(0).repeat(C, 1, 1)
    Ks = K.unsqueeze(0).repeat(C, 1, 1)
    return viewmats, Ks, width, height


@_skip_no_cuda
class TestSparsePxForward:
    def test_packed_matches_dense_reference(self):
        from gsplat.cuda._wrapper import (
            _make_lazy_cuda_func, isect_offset_encode, isect_tiles,
        )
        from gsplat.rendering import _maybe_evaluate_sh

        device = torch.device("cuda:0")
        N = 128
        C, W, H, T = 2, 64, 48, 16
        means, quats, scales, opacities, sh = _make_gaussians(N, device)
        viewmats, Ks, w, h = _make_cameras(device, C, W, H)
        ratio = 0.35
        frame, alpha, meta = _mod._sparse_px_forward(
            means, quats, scales, opacities, sh,
            viewmats, Ks, w, h, ratio,
        )
        assert frame.shape[1] == 3
        assert meta["packed"] is True
        assert meta["sampled_tile_ratio"] < 1.0

        # dense reference: same projection + SH, dense isect + rasterize
        from gsplat.cuda._wrapper import fully_fused_projection
        radii, means2d, depths, conics, _ = fully_fused_projection(
            means=means.contiguous(), covars=None, quats=quats.contiguous(),
            scales=scales.contiguous(), viewmats=viewmats.contiguous(),
            Ks=Ks.contiguous(), width=W, height=H, eps2d=0.3,
            radius_clip=0.0, packed=False, calc_compensations=False,
            camera_model="pinhole",
        )
        colors_eval = _maybe_evaluate_sh(
            _mod._SH_DEGREE, sh, means, radii, viewmats, (1,), C, N, True,
        ).contiguous()
        opac_bc = torch.broadcast_to(opacities[None, :], (C, N)).contiguous()
        tw, th = W // T, H // T
        _, isect_ids, fl_ids = isect_tiles(
            means2d, radii, depths, T, tw, th, packed=False, n_images=C,
            image_ids=None, gaussian_ids=None, conics=conics,
            opacities=opac_bc,
        )
        offs = isect_offset_encode(isect_ids, C, tw, th).reshape((1, C, th, tw))
        bg = torch.zeros(C, 3, device=device)
        rc_dense, _, _, _ = _make_lazy_cuda_func("rasterize_to_pixels_3dgs")(
            means2d.contiguous(), conics.contiguous(), colors_eval.contiguous(),
            opac_bc.contiguous(), bg, None, W, H, T, offs.contiguous(),
            fl_ids.contiguous(), False, False,
        )
        ids = meta["pixel_image_ids"]
        flat = meta["pixel_flat"]
        rr = flat % W
        # pixels order: (image_id, row, col); recover rows via flat/W
        rows = (flat // W) % H
        ref_packed = rc_dense[0][ids, rows, rr]
        assert torch.allclose(frame, ref_packed, atol=1e-5)

    def test_dense_ratio_1_returns_dense_frame(self):
        device = torch.device("cuda:0")
        N = 64
        C, W, H = 2, 64, 48
        means, quats, scales, opacities, sh = _make_gaussians(N, device)
        viewmats, Ks, w, h = _make_cameras(device, C, W, H)
        frame, alpha, meta = _mod._sparse_px_forward(
            means, quats, scales, opacities, sh,
            viewmats, Ks, w, h, 1.0,
        )
        assert tuple(frame.shape) == (1, C, H, W, 3)
        assert tuple(alpha.shape) == (1, C, H, W, 1)
        assert meta["packed"] is False

    def test_gradients_flow_to_all_params(self):
        device = torch.device("cuda:0")
        N = 64
        C, W, H = 2, 64, 48
        means, quats, scales, opacities, sh = _make_gaussians(N, device)
        for t in (means, quats, scales, opacities, sh):
            t.requires_grad_(True)
        viewmats, Ks, w, h = _make_cameras(device, C, W, H)
        ref = torch.rand(C, H, W, 3, device=device)
        frame, _, meta = _mod._sparse_px_forward(
            means, quats, scales, opacities, sh,
            viewmats, Ks, w, h, 0.35,
        )
        loss = _mod._packed_pixel_l1_loss(
            frame, ref, meta["pixel_image_ids"], meta["pixel_flat"],
        )
        loss.backward()
        assert means.grad is not None and means.grad.abs().sum().item() > 0
        assert quats.grad is not None and quats.grad.abs().sum().item() > 0
        assert scales.grad is not None and scales.grad.abs().sum().item() > 0
        assert opacities.grad is not None
        assert sh.grad is not None and sh.grad.abs().sum().item() > 0


@_skip_no_cuda
class TestSparsePxRunBackend:
    def test_run_backend_end_to_end(self):
        device = torch.device("cuda:0")
        C_train, C_eval = 2, 1
        W, H = 64, 48
        means, quats, scales, opacities, sh = _make_gaussians(80, device)
        params0 = (means, quats, scales, opacities, sh)
        viewmats, Ks, w, h = _make_cameras(device, C_train + C_eval, W, H)
        viewmats = viewmats.unsqueeze(0)  # [1, C, 4, 4]
        Ks = Ks.unsqueeze(0)
        train_idx = list(range(C_train))
        eval_idx = [C_train]
        refs_train = torch.rand(C_train, H, W, 3, device=device)
        refs_eval = torch.rand(C_eval, H, W, 3, device=device)

        class _FakeLPIPS(torch.nn.Module):
            def forward(self, x, y):
                return (x - y).pow(2).mean()

        lpips_model = _FakeLPIPS().to(device).eval()

        r = _mod.run_backend(
            "higs_sparse_px", params0, viewmats, Ks, train_idx, refs_train,
            eval_idx, refs_eval, W, H, steps=8, seed=0, device=device,
            densify_every=4, densify_threshold=5e-3, prune_threshold=0.01,
            lpips_model=lpips_model, tile_sampling_ratio=1.0,
            sampling_mode="uniform", lpips_loss_weight=0.1,
            lpips_loss_every=3, lpips_full_res=True,
            lr_decay=1.0, densify_window=None,
            pixel_raster_ratio=0.35,
        )
        assert "error" not in r
        assert r["backend"] == "higs_sparse_px"
        assert r["pixel_raster_ratio"] == 0.35
        assert 0.0 < r["sampled_tile_ratio"] < 1.0
        assert r["psnr"] > 0.0
        assert r["final_n"] > 0
        assert r["lpips_steps"] > 0  # full-res LPIPS steps ran with packed steps

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the trainable HiGS rendering path (Stage A).

Usage:
    pytest tests/experimental/render/test_trainable.py -s
"""

import pytest
import torch

device = torch.device("cuda:0")

_skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No CUDA device"
)


def _make_gaussians(N, device, sh_degree=None, seed=42):
    """Create random Gaussian parameters for testing (activated values)."""
    torch.manual_seed(seed)
    means = torch.randn(N, 3, device=device)
    means[:, 2] = means[:, 2].abs() + 2.0  # place in front of camera
    quats = torch.randn(N, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.rand(N, 3, device=device) * 0.1 + 0.01  # activated (positive)
    opacities = torch.rand(N, device=device) * 0.8 + 0.1  # activated [0.1, 0.9]

    if sh_degree is not None:
        K = (sh_degree + 1) ** 2
        colors = torch.randn(N, K, 3, device=device) * 0.1
    else:
        colors = torch.sigmoid(torch.randn(N, 3, device=device))

    return means, quats, scales, opacities, colors


def _make_camera(device, width=128, height=96):
    """Create a simple pinhole camera looking down +z."""
    focal = 128.0
    K = torch.tensor(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
    )
    viewmat = torch.eye(4, device=device)
    return viewmat, K, width, height


# ---------------------------------------------------------------------------
# Stage A: Correctness Baseline / Re-computation Proxy
# ---------------------------------------------------------------------------


class TestAPIImports:
    """Verify the Stage A API surface exists."""

    @_skip_no_cuda
    def test_trainable_function_importable(self):
        from gsplat.experimental import rasterize_gaussian_higs_trainable

        assert callable(rasterize_gaussian_higs_trainable)

    @_skip_no_cuda
    def test_trainable_function_in_render(self):
        from gsplat.experimental.render import rasterize_gaussian_higs_trainable

        assert callable(rasterize_gaussian_higs_trainable)


class TestGradModeGuards:
    """Verify that grad-mode guards work correctly."""

    @_skip_no_cuda
    def test_differentiable_false_requires_no_grad(self):
        """differentiable=False should raise if grad is enabled."""
        from gsplat.experimental import rasterize_gaussian_higs_trainable

        means, quats, scales, opacities, colors = _make_gaussians(10, device)
        viewmat, K, w, h = _make_camera(device)

        with pytest.raises(RuntimeError, match="torch.inference_mode|torch.no_grad"):
            rasterize_gaussian_higs_trainable(
                means, quats, scales, opacities, colors,
                viewmat=viewmat, K=K, width=w, height=h,
                differentiable=False,
            )

    @_skip_no_cuda
    def test_differentiable_true_allows_grad(self):
        """differentiable=True should not raise when grad is enabled."""
        from gsplat.experimental import rasterize_gaussian_higs_trainable

        means, quats, scales, opacities, colors = _make_gaussians(10, device)
        viewmat, K, w, h = _make_camera(device)

        # Should not raise RuntimeError about grad mode
        result = rasterize_gaussian_higs_trainable(
            means, quats, scales, opacities, colors,
            viewmat=viewmat, K=K, width=w, height=h,
            differentiable=True,
            sh_degree=None,
        )
        assert result.frame is not None
        assert result.frame.shape == (1, h, w, 3)

    @_skip_no_cuda
    def test_differentiable_true_rejects_inference_mode(self):
        """differentiable=True should raise if inference mode is active."""
        from gsplat.experimental import rasterize_gaussian_higs_trainable

        means, quats, scales, opacities, colors = _make_gaussians(10, device)
        viewmat, K, w, h = _make_camera(device)

        with torch.inference_mode():
            with pytest.raises(RuntimeError, match="inference mode"):
                rasterize_gaussian_higs_trainable(
                    means, quats, scales, opacities, colors,
                    viewmat=viewmat, K=K, width=w, height=h,
                    differentiable=True,
                )


class TestGradientFlow:
    """Verify gradients flow to all five parameter types."""

    @_skip_no_cuda
    def test_gradients_exist_for_all_params(self):
        """loss.backward() should produce finite, non-empty gradients on all params."""
        from gsplat.experimental import rasterize_gaussian_higs_trainable

        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        viewmat, K, w, h = _make_camera(device)

        means.requires_grad_(True)
        quats.requires_grad_(True)
        scales.requires_grad_(True)
        opacities.requires_grad_(True)
        colors.requires_grad_(True)

        result = rasterize_gaussian_higs_trainable(
            means, quats, scales, opacities, colors,
            viewmat=viewmat, K=K, width=w, height=h,
            differentiable=True,
            sh_degree=None,
        )

        loss = result.frame.sum()
        loss.backward()

        param_names = ["means", "quats", "scales", "opacities", "colors"]
        grads = [means.grad, quats.grad, scales.grad, opacities.grad, colors.grad]

        for name, grad in zip(param_names, grads):
            assert grad is not None, f"{name}.grad is None"
            assert grad.isfinite().all(), f"{name}.grad contains non-finite values"
            assert grad.abs().sum() > 0, f"{name}.grad is all zeros"

    @_skip_no_cuda
    def test_forward_output_aligns_with_standard_gsplat(self):
        """RGB/alpha outputs should approximately align with standard gsplat."""
        from gsplat.experimental import rasterize_gaussian_higs_trainable
        from gsplat.rendering import rasterization

        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        viewmat, K, w, h = _make_camera(device)

        # Run our trainable path
        result = rasterize_gaussian_higs_trainable(
            means, quats, scales, opacities, colors,
            viewmat=viewmat, K=K, width=w, height=h,
            differentiable=True,
            sh_degree=None,
        )

        # Run standard gsplat (with batch dims)
        means_b = means.unsqueeze(0)
        quats_b = quats.unsqueeze(0)
        scales_b = scales.unsqueeze(0)
        opacities_b = opacities.unsqueeze(0)
        colors_b = colors.unsqueeze(0)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
        Ks = K.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

        std_colors, std_alphas, _ = rasterization(
            means=means_b, quats=quats_b, scales=scales_b,
            opacities=opacities_b, colors=colors_b,
            viewmats=viewmats, Ks=Ks, width=w, height=h,
            packed=True,
        )

        # Compare
        torch.testing.assert_close(
            result.frame, std_colors.squeeze(0), atol=1e-3, rtol=1e-3
        )


class TestFiniteDifference:
    """Simple finite-difference gradient check on a few elements."""

    @_skip_no_cuda
    def test_finite_diff_grad_means(self):
        """Central finite-difference on means should agree with autograd."""
        from gsplat.experimental import rasterize_gaussian_higs_trainable

        N = 10
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        viewmat, K, w, h = _make_camera(device)

        # Pick one Gaussian to perturb
        idx = 3
        eps = 1e-4

        def forward_fn(m):
            m.requires_grad_(True)
            q = quats.detach().clone().requires_grad_(True)
            s = scales.detach().clone().requires_grad_(True)
            o = opacities.detach().clone().requires_grad_(True)
            c = colors.detach().clone().requires_grad_(True)
            r = rasterize_gaussian_higs_trainable(
                m, q, s, o, c,
                viewmat=viewmat, K=K, width=w, height=h,
                differentiable=True, sh_degree=None,
            )
            return r.frame.sum()

        # Autograd gradient
        m0 = means.detach().clone().requires_grad_(True)
        loss = forward_fn(m0)
        loss.backward()
        auto_grad = m0.grad[idx, 0].item()

        # Finite difference
        m_plus = means.detach().clone()
        m_plus[idx, 0] += eps
        loss_plus = forward_fn(m_plus)

        m_minus = means.detach().clone()
        m_minus[idx, 0] -= eps
        loss_minus = forward_fn(m_minus)

        fd_grad = (loss_plus.item() - loss_minus.item()) / (2 * eps)

        # Relative tolerance: finite diff is noisy, use 10%
        if abs(auto_grad) > 1e-6:
            assert abs(auto_grad - fd_grad) / max(abs(auto_grad), 1e-8) < 0.1, (
                f"Gradient mismatch for means[{idx}, 0]: auto={auto_grad:.6f}, fd={fd_grad:.6f}"
            )

    @_skip_no_cuda
    def test_finite_diff_grad_opacities(self):
        """Central finite-difference on opacities should agree with autograd."""
        from gsplat.experimental import rasterize_gaussian_higs_trainable

        N = 10
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        viewmat, K, w, h = _make_camera(device)

        idx = 3
        eps = 1e-4

        def forward_fn(o):
            m = means.detach().clone().requires_grad_(True)
            q = quats.detach().clone().requires_grad_(True)
            s = scales.detach().clone().requires_grad_(True)
            o.requires_grad_(True)
            c = colors.detach().clone().requires_grad_(True)
            r = rasterize_gaussian_higs_trainable(
                m, q, s, o, c,
                viewmat=viewmat, K=K, width=w, height=h,
                differentiable=True, sh_degree=None,
            )
            return r.frame.sum()

        o0 = opacities.detach().clone().requires_grad_(True)
        loss = forward_fn(o0)
        loss.backward()
        auto_grad = o0.grad[idx].item()

        o_plus = opacities.detach().clone()
        o_plus[idx] += eps
        loss_plus = forward_fn(o_plus)

        o_minus = opacities.detach().clone()
        o_minus[idx] -= eps
        loss_minus = forward_fn(o_minus)

        fd_grad = (loss_plus.item() - loss_minus.item()) / (2 * eps)

        if abs(auto_grad) > 1e-6:
            assert abs(auto_grad - fd_grad) / max(abs(auto_grad), 1e-8) < 0.15, (
                f"Gradient mismatch for opacities[{idx}]: auto={auto_grad:.6f}, fd={fd_grad:.6f}"
            )

    @_skip_no_cuda
    def test_finite_diff_grad_scales(self):
        """Central finite-difference on scales should agree with autograd."""
        from gsplat.experimental import rasterize_gaussian_higs_trainable

        N = 10
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        viewmat, K, w, h = _make_camera(device)

        idx = 3
        eps = 1e-4

        def forward_fn(s):
            m = means.detach().clone().requires_grad_(True)
            q = quats.detach().clone().requires_grad_(True)
            s.requires_grad_(True)
            o = opacities.detach().clone().requires_grad_(True)
            c = colors.detach().clone().requires_grad_(True)
            r = rasterize_gaussian_higs_trainable(
                m, q, s, o, c,
                viewmat=viewmat, K=K, width=w, height=h,
                differentiable=True, sh_degree=None,
            )
            return r.frame.sum()

        s0 = scales.detach().clone().requires_grad_(True)
        loss = forward_fn(s0)
        loss.backward()
        auto_grad = s0.grad[idx, 0].item()

        s_plus = scales.detach().clone()
        s_plus[idx, 0] += eps
        loss_plus = forward_fn(s_plus)

        s_minus = scales.detach().clone()
        s_minus[idx, 0] -= eps
        loss_minus = forward_fn(s_minus)

        fd_grad = (loss_plus.item() - loss_minus.item()) / (2 * eps)

        if abs(auto_grad) > 1e-6:
            assert abs(auto_grad - fd_grad) / max(abs(auto_grad), 1e-8) < 0.15, (
                f"Gradient mismatch for scales[{idx}, 0]: auto={auto_grad:.6f}, fd={fd_grad:.6f}"
            )

    @_skip_no_cuda
    def test_finite_diff_grad_colors(self):
        """Central finite-difference on colors should agree with autograd."""
        from gsplat.experimental import rasterize_gaussian_higs_trainable

        N = 10
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        viewmat, K, w, h = _make_camera(device)

        idx = 3
        eps = 1e-4

        def forward_fn(c):
            m = means.detach().clone().requires_grad_(True)
            q = quats.detach().clone().requires_grad_(True)
            s = scales.detach().clone().requires_grad_(True)
            o = opacities.detach().clone().requires_grad_(True)
            c.requires_grad_(True)
            r = rasterize_gaussian_higs_trainable(
                m, q, s, o, c,
                viewmat=viewmat, K=K, width=w, height=h,
                differentiable=True, sh_degree=None,
            )
            return r.frame.sum()

        c0 = colors.detach().clone().requires_grad_(True)
        loss = forward_fn(c0)
        loss.backward()
        auto_grad = c0.grad[idx, 0].item()

        c_plus = colors.detach().clone()
        c_plus[idx, 0] += eps
        loss_plus = forward_fn(c_plus)

        c_minus = colors.detach().clone()
        c_minus[idx, 0] -= eps
        loss_minus = forward_fn(c_minus)

        fd_grad = (loss_plus.item() - loss_minus.item()) / (2 * eps)

        if abs(auto_grad) > 1e-6:
            assert abs(auto_grad - fd_grad) / max(abs(auto_grad), 1e-8) < 0.15, (
                f"Gradient mismatch for colors[{idx}, 0]: auto={auto_grad:.6f}, fd={fd_grad:.6f}"
            )

class TestGradientCosineSimilarity:
    """Verify gradients match standard gsplat in direction (cosine similarity)."""

    @_skip_no_cuda
    def test_gradient_cosine_similarity_all_params(self):
        from gsplat.experimental import rasterize_gaussian_higs_trainable
        from gsplat.rendering import rasterization

        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        viewmat, K, w, h = _make_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)

        def _compute(use_trainable):
            m = means.detach().clone().requires_grad_(True)
            q = quats.detach().clone().requires_grad_(True)
            s = scales.detach().clone().requires_grad_(True)
            o = opacities.detach().clone().requires_grad_(True)
            c = colors.detach().clone().requires_grad_(True)
            if use_trainable:
                result = rasterize_gaussian_higs_trainable(
                    m, q, s, o, c, viewmats=viewmats, Ks=Ks,
                    width=w, height=h, differentiable=True, sh_degree=None,
                )
                result.frame.sum().backward()
            else:
                mb = m.unsqueeze(0); qb = q.unsqueeze(0); sb = s.unsqueeze(0)
                ob = o.unsqueeze(0); cb = c.unsqueeze(0)
                renders, _, _ = rasterization(
                    means=mb, quats=qb, scales=sb, opacities=ob, colors=cb,
                    viewmats=viewmats, Ks=Ks, width=w, height=h, packed=True,
                )
                renders.sum().backward()
            return {k: v.grad for k, v in [("means", m), ("quats", q), ("scales", s),
                                            ("opacities", o), ("colors", c)]}

        grads_t = _compute(True)
        grads_s = _compute(False)

        for name in ["means", "quats", "scales", "opacities", "colors"]:
            cos = torch.nn.functional.cosine_similarity(
                grads_t[name].flatten().unsqueeze(0),
                grads_s[name].flatten().unsqueeze(0),
            ).item()
            assert cos > 0.95, f"{name} cosine similarity {cos:.6f} < 0.95"
class TestTrainingSmoke:
    """End-to-end training loop smoke test."""

    @_skip_no_cuda
    def test_loss_decreases_over_steps(self):
        from gsplat.experimental import rasterize_gaussian_higs_trainable
        from gsplat.rendering import rasterization

        N = 200
        torch.manual_seed(42)
        means = torch.randn(N, 3, device=device); means[:, 2] = means[:, 2].abs() + 3.0
        quats = torch.randn(N, 4, device=device)
        quats = quats / quats.norm(dim=-1, keepdim=True)
        scales = torch.rand(N, 3, device=device) * 0.15 + 0.01
        opacities = torch.rand(N, device=device) * 0.9 + 0.05
        colors = torch.sigmoid(torch.randn(N, 3, device=device))

        viewmats = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0)
        Ks = torch.tensor([[256., 0., 128.], [0., 256., 96.], [0., 0., 1.]], device=device).unsqueeze(0).unsqueeze(0)
        W, H = 256, 192

        # Target image
        with torch.no_grad():
            target, _, _ = rasterization(
                means=means.unsqueeze(0), quats=quats.unsqueeze(0),
                scales=scales.unsqueeze(0), opacities=opacities.unsqueeze(0),
                colors=colors.unsqueeze(0), viewmats=viewmats, Ks=Ks,
                width=W, height=H, packed=True,
            )
        target_img = target.squeeze(0).squeeze(0)

        # Optimize
        m = torch.randn(N, 3, device=device).requires_grad_(True)
        q = torch.randn(N, 4, device=device)
        q.data /= q.data.norm(dim=-1, keepdim=True); q.requires_grad_(True)
        s = (torch.rand(N, 3, device=device) * 0.1 + 0.01).requires_grad_(True)
        o = (torch.rand(N, device=device) * 0.5 + 0.25).requires_grad_(True)
        c = torch.sigmoid(torch.randn(N, 3, device=device)).requires_grad_(True)

        optim = torch.optim.Adam([m, q, s, o, c], lr=0.01)
        losses = []
        for _ in range(20):
            optim.zero_grad()
            result = rasterize_gaussian_higs_trainable(
                m, q, s, o, c, viewmats=viewmats, Ks=Ks,
                width=W, height=H, differentiable=True, sh_degree=None,
            )
            loss = torch.nn.functional.mse_loss(result.frame, target_img[None])
            loss.backward()
            optim.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"
        )

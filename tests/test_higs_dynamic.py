import torch

device = torch.device("cuda:0")
from higs_skip_helpers import skipif_higs_unavailable as _skip_no_cuda

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

class TestDensifyPrune:
    @_skip_no_cuda
    def test_densify_basic(self):
        from gsplat.experimental.render.functional.gaussian_inference import _densify_gaussians
        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        grads = torch.randn(N, 3, device=device) * 0.001
        new_m, new_q, new_s, new_o, new_c = _densify_gaussians(means, quats, scales, opacities, colors, grads, threshold=0.0005)
        assert new_m.shape[0] >= N
        assert new_m.shape[1] == 3

    @_skip_no_cuda
    def test_densify_high_threshold(self):
        from gsplat.experimental.render.functional.gaussian_inference import _densify_gaussians
        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        grads = torch.zeros(N, 3, device=device)
        new_m, _, _, _, _ = _densify_gaussians(means, quats, scales, opacities, colors, grads, threshold=1.0)
        assert new_m.shape[0] == N

    @_skip_no_cuda
    def test_prune_basic(self):
        from gsplat.experimental.render.functional.gaussian_inference import _prune_gaussians
        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        opacities[:10] = 0.001
        new_m, _, _, _, _ = _prune_gaussians(means, quats, scales, opacities, colors, opacity_threshold=0.01)
        assert new_m.shape[0] == 40

    @_skip_no_cuda
    def test_prune_no_removal(self):
        from gsplat.experimental.render.functional.gaussian_inference import _prune_gaussians
        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        new_m, _, _, _, _ = _prune_gaussians(means, quats, scales, opacities, colors, opacity_threshold=0.001)
        assert new_m.shape[0] == N

class TestDynamicAPI:
    @_skip_no_cuda
    def test_training_scene_callbacks_mark_topology_dirty(self):
        from gsplat.experimental.render.functional.gaussian_inference import (
            _HIGS_DYNAMIC_SCENE,
        )

        _HIGS_DYNAMIC_SCENE.reset()
        indices = torch.tensor([0], device=device)
        for callback, args in (
            (_HIGS_DYNAMIC_SCENE.on_duplicate, (indices,)),
            (_HIGS_DYNAMIC_SCENE.on_split, (indices, indices)),
            (_HIGS_DYNAMIC_SCENE.on_remove, (indices.bool(),)),
        ):
            _HIGS_DYNAMIC_SCENE._dirty = False
            callback(*args)
            assert _HIGS_DYNAMIC_SCENE.dirty

    @_skip_no_cuda
    def test_dynamic_function_importable(self):
        from gsplat.experimental import rasterize_gaussian_higs_dynamic
        assert callable(rasterize_gaussian_higs_dynamic)

    @_skip_no_cuda
    def test_dynamic_forward_shapes(self):
        from gsplat.experimental import rasterize_gaussian_higs_dynamic
        from gsplat.experimental.render.functional.gaussian_inference import _HIGS_DYNAMIC_SCENE
        _HIGS_DYNAMIC_SCENE.reset()
        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        viewmat, K, w, h = _make_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)
        result = rasterize_gaussian_higs_dynamic(means, quats, scales, opacities, colors, viewmats=viewmats, Ks=Ks, width=w, height=h)
        assert result["frame"].shape == (1, h, w, 3)
        assert result["alpha"].shape == (1, h, w, 1)
        assert "scene_version" in result["metadata"]

    @_skip_no_cuda
    def test_dynamic_gradients_all_params(self):
        from gsplat.experimental import rasterize_gaussian_higs_dynamic
        from gsplat.experimental.render.functional.gaussian_inference import _HIGS_DYNAMIC_SCENE
        _HIGS_DYNAMIC_SCENE.reset()
        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        viewmat, K, w, h = _make_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)
        result = rasterize_gaussian_higs_dynamic(means, quats, scales, opacities, colors, viewmats=viewmats, Ks=Ks, width=w, height=h)
        result["frame"].sum().backward()
        for name, t in [("means", means), ("quats", quats), ("scales", scales), ("opacities", opacities), ("colors", colors)]:
            assert t.grad is not None
            assert t.grad.isfinite().all()
            assert t.grad.abs().sum() > 0

    @_skip_no_cuda
    def test_dynamic_forward_aligned_with_stage_b(self):
        from gsplat.experimental import rasterize_gaussian_higs_dynamic, rasterize_gaussian_higs_frozen
        from gsplat.experimental.render.functional.gaussian_inference import _HIGS_DYNAMIC_SCENE
        _HIGS_DYNAMIC_SCENE.reset()
        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        viewmat, K, w, h = _make_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)
        r_dyn = rasterize_gaussian_higs_dynamic(means, quats, scales, opacities, colors, viewmats=viewmats, Ks=Ks, width=w, height=h)
        r_frozen = rasterize_gaussian_higs_frozen(means, quats, scales, opacities, colors, viewmats=viewmats, Ks=Ks, width=w, height=h)
        torch.testing.assert_close(r_dyn["frame"], r_frozen["frame"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(r_dyn["alpha"], r_frozen["alpha"], atol=1e-5, rtol=1e-5)

class TestTopologyMutation:
    @_skip_no_cuda
    def test_densify_between_steps(self):
        from gsplat.experimental import rasterize_gaussian_higs_dynamic
        from gsplat.experimental.render.functional.gaussian_inference import _densify_gaussians, _HIGS_DYNAMIC_SCENE
        _HIGS_DYNAMIC_SCENE.reset()
        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        viewmat, K, w, h = _make_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)
        r1 = rasterize_gaussian_higs_dynamic(means, quats, scales, opacities, colors, viewmats=viewmats, Ks=Ks, width=w, height=h)
        r1["frame"].sum().backward()
        with torch.no_grad():
            new_m, new_q, new_s, new_o, new_c = _densify_gaussians(means, quats, scales, opacities, colors, means.grad, threshold=0.0001)
        _HIGS_DYNAMIC_SCENE.mark_dirty()
        for t in [new_m, new_q, new_s, new_o, new_c]:
            t.requires_grad_(True)
        r2 = rasterize_gaussian_higs_dynamic(new_m, new_q, new_s, new_o, new_c, viewmats=viewmats, Ks=Ks, width=w, height=h)
        r2["frame"].sum().backward()
        assert new_m.grad is not None
        assert r2["metadata"]["n_gaussians"] > N

    @_skip_no_cuda
    def test_prune_between_steps(self):
        from gsplat.experimental import rasterize_gaussian_higs_dynamic
        from gsplat.experimental.render.functional.gaussian_inference import _prune_gaussians, _HIGS_DYNAMIC_SCENE
        _HIGS_DYNAMIC_SCENE.reset()
        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        viewmat, K, w, h = _make_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)
        r1 = rasterize_gaussian_higs_dynamic(means, quats, scales, opacities, colors, viewmats=viewmats, Ks=Ks, width=w, height=h)
        r1["frame"].sum().backward()
        with torch.no_grad():
            new_m, new_q, new_s, new_o, new_c = _prune_gaussians(means, quats, scales, opacities, colors, opacity_threshold=0.3)
        _HIGS_DYNAMIC_SCENE.mark_dirty()
        for t in [new_m, new_q, new_s, new_o, new_c]:
            t.requires_grad_(True)
        r2 = rasterize_gaussian_higs_dynamic(new_m, new_q, new_s, new_o, new_c, viewmats=viewmats, Ks=Ks, width=w, height=h)
        r2["frame"].sum().backward()
        assert new_m.grad is not None
        assert r2["metadata"]["n_gaussians"] < N

    @_skip_no_cuda
    def test_training_topology_change_defers_pack_and_culling_rebuilds(self):
        """A dynamic training topology change must defer the packed-scene
        rebuild (the training path never consumes it) while keeping the
        non-training culling API correct: it re-packs on demand via
        ``packed_stale``."""
        from gsplat.experimental import rasterize_gaussian_higs_dynamic
        from gsplat.experimental.render.functional.gaussian_inference import (
            _HIGS_DYNAMIC_SCENE,
            _cull_gaussians_higs,
            _densify_gaussians,
        )
        _HIGS_DYNAMIC_SCENE.reset()
        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        for t_ in [means, quats, scales, opacities, colors]:
            t_.requires_grad_(True)
        viewmat, K, w, h = _make_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)
        r1 = rasterize_gaussian_higs_dynamic(
            means, quats, scales, opacities, colors,
            viewmats=viewmats, Ks=Ks, width=w, height=h,
        )
        r1["frame"].sum().backward()
        handle = _HIGS_DYNAMIC_SCENE.renderer_handle
        assert handle is not None and not handle.packed_stale
        v_before = handle.version
        with torch.no_grad():
            new_m, new_q, new_s, new_o, new_c = _densify_gaussians(
                means, quats, scales, opacities, colors, means.grad,
                threshold=0.0001,
            )
        _HIGS_DYNAMIC_SCENE.mark_dirty()
        assert handle.packed_stale, "mark_dirty should flag the packed scene stale"
        for t_ in [new_m, new_q, new_s, new_o, new_c]:
            t_.requires_grad_(True)
        r2 = rasterize_gaussian_higs_dynamic(
            new_m, new_q, new_s, new_o, new_c,
            viewmats=viewmats, Ks=Ks, width=w, height=h,
        )
        r2["frame"].sum().backward()
        # training forward deferred the pack: version advanced, packed scene
        # stays stale, n_gaussians tracks the new count.
        assert handle.version > v_before
        assert handle.packed_stale
        assert handle.n_gaussians == new_m.shape[0]
        # the non-training culling API still works and re-packs on demand.
        _, mask, _ = _cull_gaussians_higs(
            new_m, new_q, new_s, new_o, new_c,
            viewmat, K, w, h, None, renderer_handle=handle,
        )
        assert mask.shape[0] == new_m.shape[0]
        assert not handle.packed_stale, "culling API should have re-packed"
        assert handle.version > v_before
        assert handle.n_gaussians == new_m.shape[0]

    @_skip_no_cuda
    def test_multi_step_training_smoke(self):
        from gsplat.experimental import rasterize_gaussian_higs_dynamic
        from gsplat.experimental.render.functional.gaussian_inference import _densify_gaussians, _prune_gaussians, _HIGS_DYNAMIC_SCENE
        _HIGS_DYNAMIC_SCENE.reset()
        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        viewmat, K, w, h = _make_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)
        target = torch.rand(1, h, w, 3, device=device) * 0.5 + 0.25
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        opt = torch.optim.SGD([means, quats, scales, opacities, colors], lr=1.0)
        losses = []
        for step in range(10):
            opt.zero_grad()
            result = rasterize_gaussian_higs_dynamic(means, quats, scales, opacities, colors, viewmats=viewmats, Ks=Ks, width=w, height=h)
            loss = (result["frame"] - target).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if step in (3, 7):
                with torch.no_grad():
                    means, quats, scales, opacities, colors = _densify_gaussians(means, quats, scales, opacities, colors, means.grad, threshold=0.0001)
                for t in [means, quats, scales, opacities, colors]:
                    t.requires_grad_(True)
                _HIGS_DYNAMIC_SCENE.mark_dirty()
            if step == 5:
                with torch.no_grad():
                    means, quats, scales, opacities, colors = _prune_gaussians(means, quats, scales, opacities, colors, opacity_threshold=0.1)
                for t in [means, quats, scales, opacities, colors]:
                    t.requires_grad_(True)
                _HIGS_DYNAMIC_SCENE.mark_dirty()
        assert len(losses) == 10
        assert result["metadata"]["scene_version"] > 0

class TestCullCache:
    """Culling refresh-interval cache (round 37): the union-visibility
    full-N projection is reused for N forwards and invalidated by any
    topology change (densify/prune -> mark_dirty)."""

    @_skip_no_cuda
    def test_cull_cache_cadence_counts_culls(self):
        from unittest import mock
        from gsplat.experimental import rasterize_gaussian_higs_dynamic
        from gsplat.experimental.render.functional.gaussian_inference import (
            _HIGS_DYNAMIC_SCENE,
            _cull_gaussians_batched,
        )
        _HIGS_DYNAMIC_SCENE.reset()
        N = 60
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        viewmat, K, w, h = _make_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)
        calls = {"n": 0}

        def counting_cull(*args, **kwargs):
            calls["n"] += 1
            return _cull_gaussians_batched(*args, **kwargs)

        interval = 3
        import gsplat.experimental.render.functional.gaussian_inference as _g
        with mock.patch.object(_g, "_cull_gaussians_batched", counting_cull):
            metas = []
            for _ in range(5):
                res = rasterize_gaussian_higs_dynamic(
                    means, quats, scales, opacities, colors,
                    viewmats=viewmats, Ks=Ks, width=w, height=h,
                    cull_refresh_interval=interval,
                )
                metas.append(res["metadata"])
        # 5 forwards with interval 3 -> cull on forward 1 and forward 4 only.
        assert calls["n"] == 2, f"expected 2 fresh culls, got {calls['n']}"
        assert all(
            m["cull_refresh_interval"] == interval for m in metas
        )
        assert metas[-1]["n_visible"] == metas[0]["n_visible"]

    @_skip_no_cuda
    def test_cull_cache_invalidated_on_topology_change(self):
        from unittest import mock
        from gsplat.experimental import rasterize_gaussian_higs_dynamic
        from gsplat.experimental.render.functional.gaussian_inference import (
            _HIGS_DYNAMIC_SCENE,
            _cull_gaussians_batched,
            _densify_gaussians,
        )
        _HIGS_DYNAMIC_SCENE.reset()
        N = 60
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        for t in [means, quats, scales, opacities, colors]:
            t.requires_grad_(True)
        viewmat, K, w, h = _make_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)
        calls = {"n": 0}
        import gsplat.experimental.render.functional.gaussian_inference as _g
        real_cull = _g._cull_gaussians_batched

        def counting_cull(*args, **kwargs):
            calls["n"] += 1
            return real_cull(*args, **kwargs)

        with mock.patch.object(_g, "_cull_gaussians_batched", counting_cull):
            res1 = rasterize_gaussian_higs_dynamic(
                means, quats, scales, opacities, colors,
                viewmats=viewmats, Ks=Ks, width=w, height=h,
                cull_refresh_interval=1000,
            )
            res1["frame"].sum().backward()
            assert calls["n"] == 1
            with torch.no_grad():
                new_m, new_q, new_s, new_o, new_c = _densify_gaussians(
                    means, quats, scales, opacities, colors, means.grad,
                    threshold=0.0001,
                )
            _HIGS_DYNAMIC_SCENE.mark_dirty()
            for t in [new_m, new_q, new_s, new_o, new_c]:
                t.requires_grad_(True)
            res2 = rasterize_gaussian_higs_dynamic(
                new_m, new_q, new_s, new_o, new_c,
                viewmats=viewmats, Ks=Ks, width=w, height=h,
                cull_refresh_interval=1000,
            )
            res2["frame"].sum().backward()
            # densify changed the Gaussian count: the stale cache must be
            # invalidated and a fresh full-N cull must run.
            assert calls["n"] == 2, f"expected fresh cull after densify, got {calls['n']}"
            assert res2["metadata"]["n_gaussians"] > N

    @_skip_no_cuda
    def test_cull_cache_parity_static_params(self):
        """With static parameters the cached path must render identically to
        per-step culling (same visible set, same frame)."""
        from gsplat.experimental import rasterize_gaussian_higs_dynamic
        from gsplat.experimental.render.functional.gaussian_inference import (
            _HIGS_DYNAMIC_SCENE,
        )
        N = 60
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        viewmat, K, w, h = _make_camera(device)
        viewmats = viewmat.unsqueeze(0).unsqueeze(0)
        Ks = K.unsqueeze(0).unsqueeze(0)

        def run(interval):
            _HIGS_DYNAMIC_SCENE.reset()
            m, q, s, o, c = [t.detach().clone() for t in (means, quats, scales, opacities, colors)]
            res = rasterize_gaussian_higs_dynamic(
                m, q, s, o, c,
                viewmats=viewmats, Ks=Ks, width=w, height=h,
                cull_refresh_interval=interval,
            )
            return res["frame"].detach().clone(), res["metadata"]["n_visible"]

        f1, nv1 = run(1)
        f2, nv2 = run(100)
        torch.testing.assert_close(f1, f2, atol=1e-6, rtol=1e-6)
        assert nv1 == nv2
        assert nv1 > 0

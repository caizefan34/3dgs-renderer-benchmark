import sys

from pathlib import Path

path = Path("/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat/gsplat/experimental/render/functional/gaussian_inference.py")
src = path.read_text(encoding="utf-8")

def once(old, new, tag):
    global src
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"[{tag}] expected 1 occurrence, got {n}")
    src = src.replace(old, new)
    print(f"OK {tag}")

def allc(old, new, tag, expect):
    global src
    n = src.count(old)
    if n != expect:
        raise SystemExit(f"[{tag}] expected {expect} occurrences, got {n}")
    src = src.replace(old, new)
    print(f"OK {tag} ({n}x)")

def second(old, new, tag):
    global src
    n = src.count(old)
    if n != 2:
        raise SystemExit(f"[{tag}] expected 2 occurrences, got {n}")
    i = src.find(old)
    i = src.find(old, i + 1)
    src = src[:i] + new + src[i + len(old):]
    print(f"OK {tag} (2nd of 2)")

# P1: _cull_visible_cached signature
once(
    "def _cull_visible_cached(\n"
    "    handle,\n"
    "    interval: int,\n"
    "    means, quats, scales, viewmats, Ks,\n"
    "    width, height, eps2d=0.3,\n"
    "    near_plane=0.01, far_plane=1e10, radius_clip=0.0,\n"
    "    camera_model=\"pinhole\",\n"
    "):",
    "def _cull_visible_cached(\n"
    "    handle,\n"
    "    interval: int,\n"
    "    means, quats, scales, viewmats, Ks,\n"
    "    width, height, eps2d=0.3,\n"
    "    near_plane=0.01, far_plane=1e10, radius_clip=0.0,\n"
    "    camera_model=\"pinhole\",\n"
    "    cache_key: str = \"default\",\n"
    "):",
    "P1_cull_sig",
)

# P1b: docstring
once(
    "        handle: :class:`HigsRendererHandle` owning ``_cull_visible_ids`` /\n"
    "            ``_fwd_count`` state (must be non-None).",
    "        handle: :class:`HigsRendererHandle` owning the per-camera-key cull\n"
    "            cache ``_cull_cache`` (must be non-None).",
    "P1b_docstring",
)

# P2: _cull_visible_cached body
once(
    "    interval = max(1, int(interval))\n"
    "    cached = handle._cull_visible_ids\n"
    "    if (\n"
    "        cached is not None\n"
    "        and (handle._fwd_count - handle._cull_fwd_count) < interval\n"
    "    ):\n"
    "        return cached\n"
    "    vis_ids, _vis_mask, _ratio = _cull_gaussians_batched(\n"
    "        means, quats, scales, viewmats, Ks,\n"
    "        width, height,\n"
    "        eps2d=eps2d, near_plane=near_plane,\n"
    "        far_plane=far_plane, radius_clip=radius_clip,\n"
    "        camera_model=camera_model,\n"
    "    )\n"
    "    handle._cull_visible_ids = vis_ids\n"
    "    handle._cull_fwd_count = handle._fwd_count\n"
    "    return vis_ids",
    "    interval = max(1, int(interval))\n"
    "    slot = handle._cull_cache.get(cache_key)\n"
    "    if slot is not None and (handle._fwd_count - slot[1]) < interval:\n"
    "        return slot[0]\n"
    "    vis_ids, _vis_mask, _ratio = _cull_gaussians_batched(\n"
    "        means, quats, scales, viewmats, Ks,\n"
    "        width, height,\n"
    "        eps2d=eps2d, near_plane=near_plane,\n"
    "        far_plane=far_plane, radius_clip=radius_clip,\n"
    "        camera_model=camera_model,\n"
    "    )\n"
    "    handle._cull_cache[cache_key] = (vis_ids, handle._fwd_count)\n"
    "    return vis_ids",
    "P2_cull_body",
)

# P3: handle __init__
once(
    "        self._cull_visible_ids = None\n"
    "        self._cull_fwd_count: int = -1",
    "        self._cull_cache: dict = {}",
    "P3_init",
)

# P4: mark_dirty + rebuild (2x)
allc(
    "        self._cull_visible_ids = None\n"
    "        self._cull_fwd_count = -1",
    "        self._cull_cache = {}",
    "P4_dirty_rebuild",
    2,
)

# P5: autograd forward signature
once(
    "        tile_mask=None,\n"
    "        cull_refresh_interval=1,\n"
    "    ):\n"
    "        if backward_mode not in (\"higs_native\", \"gsplat_recompute\"):",
    "        tile_mask=None,\n"
    "        cull_refresh_interval=1,\n"
    "        cull_cache_key=\"default\",\n"
    "    ):\n"
    "        if backward_mode not in (\"higs_native\", \"gsplat_recompute\"):",
    "P5_fwd_sig",
)

# P6: ctx assignment
once(
    "        ctx.cull_refresh_interval = max(1, int(cull_refresh_interval))",
    "        ctx.cull_refresh_interval = max(1, int(cull_refresh_interval))\n"
    "        ctx.cull_cache_key = cull_cache_key",
    "P6_ctx",
)

# P7: cull call cache_key
once(
    "                    vis_ids = _cull_visible_cached(\n"
    "                        renderer_handle, ctx.cull_refresh_interval,\n"
    "                        means, quats, scales, viewmats, Ks,\n"
    "                        width, height,\n"
    "                        eps2d=eps2d, near_plane=near_plane,\n"
    "                        far_plane=far_plane, radius_clip=radius_clip,\n"
    "                        camera_model=camera_model,\n"
    "                    )",
    "                    vis_ids = _cull_visible_cached(\n"
    "                        renderer_handle, ctx.cull_refresh_interval,\n"
    "                        means, quats, scales, viewmats, Ks,\n"
    "                        width, height,\n"
    "                        eps2d=eps2d, near_plane=near_plane,\n"
    "                        far_plane=far_plane, radius_clip=radius_clip,\n"
    "                        camera_model=camera_model,\n"
    "                        cache_key=ctx.cull_cache_key,\n"
    "                    )",
    "P7_cull_call",
)

# P8: _higs_dynamic_forward signature
once(
    "    tile_mask=None,\n"
    "    cull_refresh_interval: int = 1,\n"
    "    scene=None,\n"
    ") -> dict:\n"
    "    \"\"\"Forward pass for dynamic-topology HiGS diff. path (Stage C).",
    "    tile_mask=None,\n"
    "    cull_refresh_interval: int = 1,\n"
    "    cull_cache_key: str = \"default\",\n"
    "    scene=None,\n"
    ") -> dict:\n"
    "    \"\"\"Forward pass for dynamic-topology HiGS diff. path (Stage C).",
    "P8_dynfwd_sig",
)

# P9: dynamic apply call
once(
    "        sampling_mode, tile_mask, cull_refresh_interval,\n"
    "    )\n"
    "\n"
    "    scene_version = _HIGS_DYNAMIC_SCENE.next_version(means.shape[0])",
    "        sampling_mode, tile_mask, cull_refresh_interval,\n"
    "        cull_cache_key,\n"
    "    )\n"
    "\n"
    "    scene_version = _HIGS_DYNAMIC_SCENE.next_version(means.shape[0])",
    "P9_dyn_apply",
)

# P10: rasterize_gaussian_higs_dynamic wrapper signature
once(
    "    tile_mask=None,\n"
    "    cull_refresh_interval: int = 1,\n"
    "    scene=None,\n"
    ") -> dict:\n"
    "    \"\"\"Differentiable HiGS rendering with dynamic-topology support (Stage C).",
    "    tile_mask=None,\n"
    "    cull_refresh_interval: int = 1,\n"
    "    cull_cache_key: str = \"default\",\n"
    "    scene=None,\n"
    ") -> dict:\n"
    "    \"\"\"Differentiable HiGS rendering with dynamic-topology support (Stage C).",
    "P10_dynwrap_sig",
)

# P11: dynamic wrapper call to _higs_dynamic_forward (2nd of 2 identical blocks)
second(
    "        cull_refresh_interval=cull_refresh_interval,\n"
    "        scene=scene,\n"
    "    )",
    "        cull_refresh_interval=cull_refresh_interval,\n"
    "        cull_cache_key=cull_cache_key,\n"
    "        scene=scene,\n"
    "    )",
    "P11_dynwrap_call",
)

# P12: _assemble_grads must return one gradient slot per forward input
# (cull_cache_key was added to forward but not to the backward tuple; without
# this the autograd engine raises "incorrect number of gradients").
once(
    "            None,  # tile_mask (bool tensor, non-differentiable input)\n"
    "            None,  # cull_refresh_interval (non-tensor input)\n"
    "        )",
    "            None,  # tile_mask (bool tensor, non-differentiable input)\n"
    "            None,  # cull_refresh_interval (non-tensor input)\n"
    "            None,  # cull_cache_key (non-tensor input)\n"
    "        )",
    "P12_assemble_grads",
)
path.write_text(src, encoding="utf-8")
print("PATCHED_ALL_OK")
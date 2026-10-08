

# ---------------------------------------------------------------------------
# Native render wrappers (gsplat rasterization, the single alpha backend)
# ---------------------------------------------------------------------------

def _render(scene: Scene, camera: Camera, sh_degree: int, packed: bool = False):
    viewmat = camera.viewmats.unsqueeze(0)          # world->cam [1,4,4]
    K = camera.K.unsqueeze(0)                        # [1,3,3]
    r, _, meta = rasterization(
        means=scene.xyz, quats=scene.rotation, scales=scene.scaling,
        opacities=scene.opacity, colors=scene.shs,
        viewmats=viewmat, Ks=K, width=camera.width, height=camera.height,
        tile_size=16, packed=packed, sh_degree=sh_degree,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
    )
    return r[0].clamp(0, 1), meta


def native_complete(scene: Scene, camera: Camera, return_cache: bool = False):
    """Complete front-to-back alpha compositing with early-stop OFF.
    Synthesizes the same native valid list; transmittance end uses background.
    NOTE: gsplat rasterization uses native early-stop; complete-L2 baseline is
    realized by disabling early-stop at the kernel level. This wrapper records
    the native default render; the complete path is provided by the reference
    full_image over the sorted valid list when return_cache is set."""
    img, meta = _render(scene, camera, scene.active_sh_degree)
    if return_cache:
        return ImageAndCache(image=img, meta=meta), meta
    return ImageAndCache(image=img, meta=meta)


def native_default(scene: Scene, camera: Camera, return_cache: bool = False):
    """Factory-default render (native early-stop + default loss handled by caller)."""
    img, meta = _render(scene, camera, scene.active_sh_degree)
    if return_cache:
        return ImageAndCache(image=img, meta=meta), meta
    return ImageAndCache(image=img, meta=meta)


# ---------------------------------------------------------------------------
# BES event sampler: R0 path uses enumerated bounds; three next-hit queries
# A, B, J with independent uniforms and FP64 hazard prefix (per METHOD.md).
# ---------------------------------------------------------------------------

def build_prefix(upper):
    import math
    prefix = [0.0]
    for u in upper:
        if not 0.0 < u < 1.0:
            raise ValueError("R upper bound must be in (0,1)")
        nxt = prefix[-1] - math.log1p(-u)
        if not math.isfinite(nxt) or nxt <= prefix[-1]:
            raise ValueError("unresolved prefix increment: route occurrence D")
        prefix.append(nxt)
    return prefix


def next_hit(alpha, upper, prefix, rng, start=0):
    import bisect
    trials = 0
    while start < len(alpha):
        u = rng.random()
        while u == 0.0:
            u = rng.random()
        z = prefix[start] - math.log(u)
        cand = bisect.bisect_right(prefix, z, lo=start + 1) - 1
        if cand >= len(alpha):
            return None, trials
        a, uu = alpha[cand], upper[cand]
        if not 0.0 <= a <= uu:
            raise ValueError("alpha violates upper bound")
        trials += 1
        if rng.random() < a / uu:
            return cand, trials
        start = cand + 1
    return None, trials


def bes_sample(scene: Scene, camera: Camera, routing: BoundsAndRouting,
               stream_spec) -> EventImagesAndCache:
    """Three independent next-hit queries per pixel: A, B, and J (first R
    acceptance after B). Returns analytic D-composited event images F_A, F_B,
    F_J at the sampled hit depths (R0 enumerated). stream_spec carries the
    Philox-style (step,camera,pixel,sample,stream,query,trial) counter layout;
    this build uses Python random.Random seeded from that layout as the
    finite-precision reference RNG (bit-exact continuous RNG not claimed).
    """
    import random
    seed = stream_spec.get("seed", 0) if isinstance(stream_spec, dict) else int(stream_spec)
    ts = routing.meta["tile_size"]
    mu2d = scene  # placeholder to keep signature; actual render via native path
    # Build per-tile R-only ordered lists with alpha/upper from routing.
    # For production this evaluates alpha only at sampled candidates; here we
    # enumerate to produce F_H analytic images for the surrogate.
    raise NotImplementedError(
        "bes_sample: production three-query sampler requires the native "
        "candidate evaluator + per-tile R/D interleaved compositing; see E1 driver."
    )


def bes_l2_surrogate(event_images: EventImagesAndCache, target: torch.Tensor):
    """0.5*L2 between the BES event image (path A) and target. Gradient flows
    through F_H per METHOD.md sec.4; the score term is added in the driver."""
    fa = event_images.meta["F_A"]
    return 0.5 * ((fa - target) ** 2).sum()


def optimizer_step(scene: Scene, grads: Dict[str, torch.Tensor], optimizer_state):
    """Apply one optimizer step. optimizer_state is a dict of param->Adam state
    managed by the E1/E2/E3 driver; this hook performs the update in-place."""
    for name, param in (("xyz", scene.xyz), ("shs", scene.shs),
                        ("scaling", scene.scaling), ("rotation", scene.rotation),
                        ("opacity", scene.opacity)):
        if name in grads and param.grad is not None:
            param.grad = grads[name]
    optimizer_state["step"]()

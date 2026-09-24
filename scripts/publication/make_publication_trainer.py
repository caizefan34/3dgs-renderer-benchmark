#!/usr/bin/env python3
"""Create publication_trainer.py from final30k_trainer.py on mx via asserted
textual replacements. Every replacement must match exactly once or the script
fails loudly (no silent partial patches)."""
import sys

SRC = "/home/liaoyuanjun/3dgs-renderer-benchmark/final30k_trainer.py"
DST = "/home/liaoyuanjun/3dgs-renderer-benchmark/publication_trainer.py"

text = open(SRC, encoding="utf-8").read()
n0 = len(text)


def rep(old, new):
    global text
    assert text.count(old) == 1, f"anchor not unique ({text.count(old)}): {old[:80]!r}"
    text = text.replace(old, new)


# P1: header
rep('''"""FINAL-30K unified trainer: reference_v1 frozen recipe, two renderer arms.

  --arm b1a : gsplat true-accutile rasterization (absgrad=True, accutile=True, eps2d=0.1)
              via the matched-env B1A extension (gsplat_cuda_final30k, sys.modules-injected)
  --arm c0  : C0_V3_FINAL30K dynamic forward (F9 + SCALAR_ADJOINT + H8_MR, PX=2,
              eps2d=0.3, HIGS_BWD_ABSGRAD=1) via the 9baf8655 extension
''',
    '''"""PUBLICATION unified trainer: reference_v1 frozen recipe, multi-arm.

  --arm b1a : gsplat true-accutile rasterization (absgrad=True, accutile=True, eps2d=0.1)
              via the matched-env B1A extension (gsplat_cuda_final30k, sys.modules-injected)
  --arm b1  : B1_CLEAN_GSPLAT: pristine upstream gsplat v1.5.3 (937e2991), same call
              as b1a minus accutile (absgrad is native upstream), eps2d=0.1 default
  --arm c0  : C0_V3_FINAL30K dynamic forward (F9 + SCALAR_ADJOINT + H8_MR, PX=2,
              eps2d=0.3, HIGS_BWD_ABSGRAD=1) via the 9baf8655 extension
  --arm a0  : ablation base = the internal parent: gather-based forward (F9 disabled),
              tensor blend adjoint, no H8-MR -- SAME 9baf8655 binary, env-switched
  --arm a1  : a0 + F9 (gatherless producer on, tensor adjoint, no H8-MR)
  --arm a2  : a1 + SCALAR_ADJOINT (scalar adjoint on, no H8-MR)
  --eps2d O : override the arm's eps2d (P4 sensitivity; runtime kwarg, no rebuild)
''')

# P2: B1 paths
rep('SCENE_CUDA_DIR = "/mnt/storage_pool/liaoyuanjun/torchext/gsplat_scene_cuda"',
    '''SCENE_CUDA_DIR = "/mnt/storage_pool/liaoyuanjun/torchext/gsplat_scene_cuda"
B1_TREE = "/mnt/storage_pool/liaoyuanjun/gsplat-b1-clean-v153"
B1_SO = "/mnt/storage_pool/liaoyuanjun/gsplat_b1clean_pub_cache/gsplat_cuda_b1clean/gsplat_cuda_b1clean.so"''')

# P3: new bootstrap functions before main
rep('''# ================================================================ main
def main():''',
    '''def bootstrap_b1():
    mod = load_ext_module("gsplat_cuda_b1clean", B1_SO)
    sys.modules["gsplat.csrc"] = mod
    sys.path.insert(0, B1_TREE)
    sys.path.insert(0, PHASE7_DIR)
    sys.path.insert(0, SRC_DIR)
    sys.path.insert(0, REFERENCE_V1_DIR)
    import gsplat
    import inspect
    sig = inspect.signature(gsplat.rasterization)
    assert "accutile" not in sig.parameters, "accutile present -- not the pristine tree"
    assert "absgrad" in sig.parameters, "absgrad missing -- wrong tree"
    return {"arm": "b1", "so_sha256": sha256_file(B1_SO), "so_path": B1_SO,
            "gsplat_file": gsplat.__file__,
            "base_commit": "937e29912570c372bed6747a5c9bf85fed877bae (v1.5.3 pristine)"}


# A0 = internal parent (gather forward, tensor adjoint, no H8-MR).
# The three modules are RUNTIME env switches in the SAME frozen 9baf8655 binary.
ABLATION_ENV = {
    "a0": {"HIGS_DISABLE_F9": "1"},
    "a1": {},
    "a2": {"HIGS_BWD_SCALAR_ADJOINT": "scalar_adjoint"},
}


def bootstrap_a(arm):
    for k in ("HIGS_PX_RUNTIME", "HIGS_BWD_SCALAR_ADJOINT", "HIGS_BWD_H8_MR",
              "HIGS_BWD_ABSGRAD", "HIGS_DISABLE_F9"):
        os.environ.pop(k, None)
    os.environ["HIGS_PX_RUNTIME"] = "2"          # frozen substrate: launch shape only
    os.environ["HIGS_BWD_ABSGRAD"] = "1"         # densification statistic (pure aux)
    os.environ.update(ABLATION_ENV[arm])
    sys.path.insert(0, C0_WT)
    sys.path.insert(0, SCENE_CUDA_DIR)
    core = load_ext_module("gsplat_cuda", CORE_SO)
    sys.modules["gsplat.csrc"] = core
    exp = load_ext_module("experimental_gaussian_render_inference_scene_cuda", C0_SO)
    sys.modules["gsplat.experimental.render.kernels.csrc"] = exp
    sys.modules["experimental_gaussian_render_inference_scene_cuda"] = exp
    sys.path.insert(0, PHASE7_DIR)
    sys.path.insert(0, SRC_DIR)
    sys.path.insert(0, REFERENCE_V1_DIR)
    from gsplat.experimental import rasterize_gaussian_higs_dynamic  # noqa: F401
    from gsplat.experimental.render.functional.gaussian_inference import _HIGS_DYNAMIC_SCENE  # noqa: F401
    return {"arm": arm, "so_sha256": sha256_file(C0_SO), "so_path": C0_SO,
            "core_sha256": sha256_file(CORE_SO),
            "env": {k: os.environ.get(k) for k in
                    ("HIGS_PX_RUNTIME", "HIGS_BWD_SCALAR_ADJOINT", "HIGS_BWD_H8_MR",
                     "HIGS_BWD_ABSGRAD", "HIGS_DISABLE_F9")},
            "binary_same_as": "c0 (ext 9baf8655 + core 361b216b)"}


# ================================================================ main
def main():''')

# P4: argparse
rep('    ap.add_argument("--arm", required=True, choices=["b1a", "c0"])',
    '''    ap.add_argument("--arm", required=True,
                    choices=["b1a", "b1", "c0", "a0", "a1", "a2"])
    ap.add_argument("--eps2d", type=float, default=None,
                    help="override the arm's eps2d (P4 sensitivity; runtime kwarg)")''')

# P5: bootstrap dispatch
rep('''    if args.arm == "b1a":
        ident = bootstrap_b1a()
    else:
        ident = bootstrap_c0()
''',
    '''    if args.arm == "b1a":
        ident = bootstrap_b1a()
    elif args.arm == "b1":
        ident = bootstrap_b1()
    elif args.arm in ("a0", "a1", "a2"):
        ident = bootstrap_a(args.arm)
    else:
        ident = bootstrap_c0()

    eps2d_eff = args.eps2d if args.eps2d is not None else (
        0.1 if args.arm in ("b1a", "b1") else 0.3)
''')

# P6: import dispatch
rep('''    if args.arm == "b1a":
        from gsplat import rasterization
    else:''',
    '''    if args.arm in ("b1a", "b1"):
        from gsplat import rasterization
    else:''')

# P7a: render call (gsplat arms)
rep('''        if args.arm == "b1a":
            r, _, meta = rasterization(
                means=model.get_xyz, quats=model.get_rotation,
                scales=model.get_scaling, opacities=model.get_opacity,
                colors=model.get_features,
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=16, packed=False, sh_degree=sh_degree,
                radius_clip=0.0, eps2d=0.1, render_mode="RGB",
                absgrad=True, accutile=True)''',
    '''        if args.arm in ("b1a", "b1"):
            # accutile exists ONLY in the B1A patched tree; the pristine v1.5.3
            # signature has no such kwarg, so it must be omitted entirely for b1
            extra = {"accutile": True} if args.arm == "b1a" else {}
            r, _, meta = rasterization(
                means=model.get_xyz, quats=model.get_rotation,
                scales=model.get_scaling, opacities=model.get_opacity,
                colors=model.get_features,
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=16, packed=False, sh_degree=sh_degree,
                radius_clip=0.0, eps2d=eps2d_eff, render_mode="RGB",
                absgrad=True, **extra)''')

# P7b: render call (higs arms) -- pass eps2d explicitly
rep('''                backward_mode="higs_native", enable_culling=True,
                camera_model="pinhole", render_mode="RGB")  # eps2d default 0.3 (frozen C0)''',
    '''                backward_mode="higs_native", enable_culling=True,
                camera_model="pinhole", render_mode="RGB",
                eps2d=eps2d_eff)''')

# P8: three c0 guards -> HIGS arms
rep('        if args.arm == "c0" and iteration <= config.densify_until_iter:',
    '        if args.arm in ("c0", "a0", "a1", "a2") and iteration <= config.densify_until_iter:')
rep('''            if args.arm == "c0":
                _HIGS_DYNAMIC_SCENE.mark_dirty()  # topology changed''',
    '''            if args.arm in ("c0", "a0", "a1", "a2"):
                _HIGS_DYNAMIC_SCENE.mark_dirty()  # topology changed''')
rep('''            if args.arm == "c0":
                _HIGS_DYNAMIC_SCENE.mark_dirty()  # opacity is a packed-buffer value''',
    '''            if args.arm in ("c0", "a0", "a1", "a2"):
                _HIGS_DYNAMIC_SCENE.mark_dirty()  # opacity is a packed-buffer value''')

# P9: renderer metadata
rep('''        "renderer": {
            "b1a": {"eps2d": 0.1, "absgrad": True, "accutile": True, "tile_size": 16, "packed": False},
            "c0": {"eps2d": 0.3, "env": ident.get("env"), "backward_mode": "higs_native",
                    "f9": True, "scalar_adjoint": True, "h8_mr": True, "px_runtime": 2},
        }[args.arm],''',
    '''        "renderer": {
            "b1a": {"eps2d": eps2d_eff, "absgrad": True, "accutile": True, "tile_size": 16, "packed": False},
            "b1": {"eps2d": eps2d_eff, "absgrad": True, "accutile": False, "tile_size": 16, "packed": False,
                    "base_commit": "937e29912570c372bed6747a5c9bf85fed877bae (v1.5.3 pristine)"},
            "c0": {"eps2d": eps2d_eff, "env": ident.get("env"), "backward_mode": "higs_native",
                    "f9": True, "scalar_adjoint": True, "h8_mr": True, "px_runtime": 2},
            "a0": {"eps2d": eps2d_eff, "env": ident.get("env"), "backward_mode": "higs_native",
                    "f9": False, "scalar_adjoint": False, "h8_mr": False, "px_runtime": 2},
            "a1": {"eps2d": eps2d_eff, "env": ident.get("env"), "backward_mode": "higs_native",
                    "f9": True, "scalar_adjoint": False, "h8_mr": False, "px_runtime": 2},
            "a2": {"eps2d": eps2d_eff, "env": ident.get("env"), "backward_mode": "higs_native",
                    "f9": True, "scalar_adjoint": True, "h8_mr": False, "px_runtime": 2},
        }[args.arm],
        "eps2d_override": args.eps2d,''')

# ---------------------------------------------------------------- B0 arm
# P10: header docstring b0 line
rep('''  --eps2d O : override the arm's eps2d (P4 sensitivity; runtime kwarg, no rebuild)
''',
    '''  --arm b0  : ORIGINAL graphdeco 3DGS @ 54c035f (original diff_gaussian_rasterization,
              ORIGINAL_METHOD_PROTOCOL densification: signed 2D-grad L2 norm >= 0.0002,
              pixel-space, no rescale)
  --eps2d O : override the arm's eps2d (P4 sensitivity; runtime kwarg, no rebuild)
''')

# P11: B0 path constant
rep('B1_SO = "/mnt/storage_pool/liaoyuanjun/gsplat_b1clean_pub_cache/gsplat_cuda_b1clean/gsplat_cuda_b1clean.so"',
    '''B1_SO = "/mnt/storage_pool/liaoyuanjun/gsplat_b1clean_pub_cache/gsplat_cuda_b1clean/gsplat_cuda_b1clean.so"
B0_RASTER_DIR = "/mnt/storage_pool/liaoyuanjun/graphdeco-b0-pub/submodules/diff-gaussian-rasterization"''')

# P12: bootstrap_b0 (before the A0 comment block)
rep('''# A0 = internal parent (gather forward, tensor adjoint, no H8-MR).''',
    '''def bootstrap_b0():
    """B0 = ORIGINAL_METHOD_PROTOCOL: graphdeco 3DGS @ 54c035f, original
    diff_gaussian_rasterization, original densification statistic."""
    sys.path.insert(0, B0_RASTER_DIR)
    sys.path.insert(0, PHASE7_DIR)
    sys.path.insert(0, SRC_DIR)
    sys.path.insert(0, REFERENCE_V1_DIR)
    from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
    return {"arm": "b0", "package_dir": B0_RASTER_DIR,
            "base_commit": "54c035f7834b564019656c3e3fcc3646292f727d "
                           "(graphdeco HEAD; ORIGINAL_METHOD_PROTOCOL)"}


# A0 = internal parent (gather forward, tensor adjoint, no H8-MR).''')

# P13: argparse choices += b0
rep('choices=["b1a", "b1", "c0", "a0", "a1", "a2"])',
    'choices=["b1a", "b1", "b0", "c0", "a0", "a1", "a2"])')

# P14: bootstrap dispatch += b0; eps2d_eff None for b0
rep('''    elif args.arm == "b1":
        ident = bootstrap_b1()
    elif args.arm in ("a0", "a1", "a2"):
        ident = bootstrap_a(args.arm)
    else:
        ident = bootstrap_c0()

    eps2d_eff = args.eps2d if args.eps2d is not None else (
        0.1 if args.arm in ("b1a", "b1") else 0.3)
''',
    '''    elif args.arm == "b1":
        ident = bootstrap_b1()
    elif args.arm == "b0":
        ident = bootstrap_b0()
    elif args.arm in ("a0", "a1", "a2"):
        ident = bootstrap_a(args.arm)
    else:
        ident = bootstrap_c0()

    # b0 has no eps2d concept (original rasterizer); None is never consumed by it
    eps2d_eff = args.eps2d if args.eps2d is not None else (
        0.1 if args.arm in ("b1a", "b1") else
        (0.3 if args.arm in ("c0", "a0", "a1", "a2") else None))
''')

# P15: import dispatch += b0
rep('''    if args.arm in ("b1a", "b1"):
        from gsplat import rasterization
    else:''',
    '''    if args.arm in ("b1a", "b1"):
        from gsplat import rasterization
    elif args.arm == "b0":
        from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
    else:''')

# P16: render_with_meta b0 branch (first branch, before the gsplat branch)
rep('''        if args.arm in ("b1a", "b1"):
            # accutile exists ONLY in the B1A patched tree; the pristine v1.5.3''',
    '''        if args.arm == "b0":
            # ORIGINAL graphdeco render path (gaussian_renderer/__init__.py @ 54c035f):
            # settings from the graphdeco-convention camera fields the GTDataset
            # already exposes (world_view_transform = W2C.T, full_proj = (P@W2C).T).
            raster_settings = GaussianRasterizationSettings(
                image_height=int(cam.image_height), image_width=int(cam.image_width),
                tanfovx=float(cam.tanfovx), tanfovy=float(cam.tanfovy),
                bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
                viewmatrix=cam.world_view_transform,
                projmatrix=cam.full_proj_transform,
                sh_degree=sh_degree, campos=cam.camera_center,
                prefiltered=False, debug=False, antialiasing=False)
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            means2d = torch.zeros_like(model.get_xyz, requires_grad=True, device="cuda")
            image, radii, _depth = rasterizer(
                means3D=model.get_xyz, means2D=means2d, shs=model.get_features,
                colors_precomp=None,
                # benchmark model stores opacity 1-D [N] (gsplat convention); the
                # original rasterizer's backward returns [N,1] (graphdeco
                # convention) -- unsqueeze makes the adjoint shapes consistent
                opacities=model.get_opacity.unsqueeze(-1),
                scales=model.get_scaling, rotations=model.get_rotation,
                cov3D_precomp=None)
            frame = image.permute(1, 2, 0)
            # original radii is [N] 1-D; reshape to [1, N, 1] so the trainer's
            # generic (radii > 0).any(dim=-1) yields a proper [N] mask (a 1-D
            # radii would collapse any() to a 0-D bool and break downstream
            # boolean indexing)
            return frame.clamp(0, 1), radii.reshape(1, -1, 1), means2d
        if args.arm in ("b1a", "b1"):
            # accutile exists ONLY in the B1A patched tree; the pristine v1.5.3''')

# P17: densification statistic call -- b0 = official signed grad, no rescale
rep('''            model.add_densification_stats(means2d, visibility_filter,
                                          width=cam.image_width, height=cam.image_height)''',
    '''            if args.arm == "b0":
                # ORIGINAL_METHOD_PROTOCOL: signed 2D-grad L2 norm in pixel space,
                # no rescale (the original rasterizer's means2D.grad is already
                # pixel-space); the model falls back to .grad because the original
                # means2D tensor has no absgrad attribute.
                model.add_densification_stats(means2d, visibility_filter)
            else:
                model.add_densification_stats(means2d, visibility_filter,
                                              width=cam.image_width, height=cam.image_height)''')

# P18: densification threshold -- b0 = official 0.0002
rep('''        if (iteration > config.densify_from_iter and''',
    '''        # B0 = ORIGINAL_METHOD_PROTOCOL densification threshold (0.0002, signed);
        # every other arm uses the frozen matched-protocol threshold (absgrad, 0.0008).
        thresh_eff = 0.0002 if args.arm == "b0" else config.densify_grad_threshold
        if (iteration > config.densify_from_iter and''')
rep('                selected = gnorm >= config.densify_grad_threshold',
    '                selected = gnorm >= thresh_eff')
rep('''            ev = model.densify_and_prune(
                max_grad=config.densify_grad_threshold,''',
    '''            ev = model.densify_and_prune(
                max_grad=thresh_eff,''')

# P19: renderer metadata += b0
rep('''            "b1": {"eps2d": eps2d_eff, "absgrad": True, "accutile": False, "tile_size": 16, "packed": False,
                    "base_commit": "937e29912570c372bed6747a5c9bf85fed877bae (v1.5.3 pristine)"},''',
    '''            "b1": {"eps2d": eps2d_eff, "absgrad": True, "accutile": False, "tile_size": 16, "packed": False,
                    "base_commit": "937e29912570c372bed6747a5c9bf85fed877bae (v1.5.3 pristine)"},
            "b0": {"renderer": "diff_gaussian_rasterization @ 54c035f (original)",
                    "densify_stat": "signed_grad_norm_pixel_space",
                    "densify_threshold": 0.0002, "absgrad": False,
                    "antialiasing": False, "scale_modifier": 1.0,
                    "eps2d": None},''')

# P20: --seed for P6 multi-seed confirmation (default None = frozen seed 42)
rep('''    ap.add_argument("--final-eval", choices=["all", "subset"], default="all")''',
    '''    ap.add_argument("--seed", type=int, default=None,
                    help="override the frozen seed 42 (P6 multi-seed confirmation)")
    ap.add_argument("--final-eval", choices=["all", "subset"], default="all")''')
rep('''    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)''',
    '''    seed_val = args.seed if args.seed is not None else 42
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)''')
rep('''    config = ReferenceV1Config()''',
    '''    config = ReferenceV1Config()
    if args.seed is not None:
        config.seed = args.seed  # camera-sequence rng + recorded metadata follow''')

open(DST, "w", encoding="utf-8").write(text)
print(f"patched OK: {n0} -> {len(text)} bytes")
print(f"WROTE {DST}")

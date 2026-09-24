#!/usr/bin/env python3
"""P3-H: backward accumulation ceiling diagnostic for the frozen 3DGS native
backward kernel.

Builds 5 compile-time variants (V0..V4) of the diagnostic gsplat worktree
(``higs_p3h_worktree_gsplat``) that differ ONLY in how the per-Gaussian
gradient atomic-scatter block of the pixel-blend backward behaves:

  V0  P3=0  FULL                  (frozen C0 V3 behavior; real atomics B+C+D)
  V1  P3=1  ACCUM_SINK_ALL        (sinks B color + C opacity + D H8 geom)
  V2  P3=2  ACCUM_SINK_GEOM       (sinks D geometry only)
  V3  P3=3  ACCUM_SINK_APPEAR     (sinks B + C appearance only)
  V4  P3=4  WRITE_ONLY/NO_ATOMIC  (full values, plain uncontended stores)

Each variant is built via torch ``cpp_extension`` (``build.py``) into an
isolated cache dir ``<TORCH_EXTENSIONS_DIR>/V{n}`` with
``NVCC_FLAGS=-DP3_SINK_MODE={n}``.

Subcommands (run ON ``mx`` with the torch env active):
  build [--variant Vn|--all]   build variant(s)
  hash                          hash built .so + source -> binary_hashes.json,
                                provenance.json
  emit-static                   write backward_semantic_partition.json,
                                accumulation_sites.json, variant_definitions.json
  smoke                         run the native backward ONCE per variant on the
                                room fixture at 2048; assert finite; report
                                PASS/FAIL + backward/forward ms (informational,
                                DIAGNOSTIC_SHARED_GPU)
  timing                        full balanced timing protocol (implemented;
                                normally run later on an idle GPU)
"""
import argparse, hashlib, importlib, importlib.util, json, math, os, subprocess
import sys, time, types, uuid
from pathlib import Path

try:
    import numpy as np
    import torch
    _HAS_TORCH = True
except Exception:  # allow build-only contexts that don't need CUDA torch
    torch = None
    np = None
    _HAS_TORCH = False

# --- paths (fixed for the mx box) -------------------------------------------
WORK = "/mnt/storage_pool/liaoyuanjun/higs_p3h_worktree_gsplat"      # diagnostic worktree (patched)
FROZEN = "/mnt/storage_pool/liaoyuanjun/higs_h8_mr_worktree"         # frozen production worktree (DO NOT TOUCH)
CACHE_BASE = "/mnt/storage_pool/liaoyuanjun/higs_p3h_cache"
REMOTE_ART = os.path.join(CACHE_BASE, "artifacts")
EXT_NAME = "experimental_gaussian_render_inference_scene_cuda"
SINK_REL = "experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu"
BUILD_PY_REL = "experimental/render/kernels/cuda/build.py"
ENV_BIN = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin"

DIAG_SHA = "f01b624b99d7ed2ae33359f1f5f2bccd95e4c8b73955471f3b40e6571a837665"
FROZEN_SHA_PREFIX = "e657485f9a187be6"

TILE_SIZE = 16
SH_DEGREE = 3
MAX_LONG_SIDE = 2048
EPS2D = 0.3
NEAR, FAR, RADIUS_CLIP = 0.01, 1e10, 0.0
CAMERA_MODEL = "pinhole"
RENDER_MODE = "RGB"
GPU_TAG = "DIAGNOSTIC_SHARED_GPU"

# HIGS variant registry -------------------------------------------------------
VARIANTS = {
    "V0": {"mode": "FULL",              "P3": 0, "name": "V0 FULL C0-V3",
           "sunk": [],                  "real": ["B color", "C opacity", "D H8 geom"]},
    "V1": {"mode": "ACCUM_SINK_ALL",    "P3": 1, "name": "V1 ACCUM_SINK_ALL",
           "sunk": ["B color", "C opacity", "D H8 geom"], "real": []},
    "V2": {"mode": "ACCUM_SINK_GEOM",   "P3": 2, "name": "V2 ACCUM_SINK_GEOM",
           "sunk": ["D H8 geom"],       "real": ["B color", "C opacity"]},
    "V3": {"mode": "ACCUM_SINK_APPEAR", "P3": 3, "name": "V3 ACCUM_SINK_APPEAR",
           "sunk": ["B color", "C opacity"], "real": ["D H8 geom"]},
    "V4": {"mode": "WRITE_ONLY_NO_ATOMIC", "P3": 4, "name": "V4 WRITE_ONLY",
           "sunk": ["B color", "C opacity", "D H8 geom (plain stores)"], "real": []},
}

SCENE_CONFIGS = {
    "room": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json",
        "native_w": 3114, "native_h": 2075,
    },
    "bicycle": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/bicycle/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/bicycle/cameras.json",
        "native_w": 4946, "native_h": 3286,
    },
    "garden": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/garden/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/garden/cameras.json",
        "native_w": 5187, "native_h": 3361,
    },
}


def so_path_for(variant):
    return os.path.join(CACHE_BASE, variant, EXT_NAME, EXT_NAME + ".so")


def build_meta_path(variant):
    return os.path.join(CACHE_BASE, variant, "build_meta.json")


def _set_runtime_env():
    os.environ["HIGS_BWD_SCALAR_ADJOINT"] = "scalar_adjoint"  # frozen C0 V3 adjoint
    os.environ["HIGS_BWD_H8_MR"] = "1"                          # H8 moment reduction on
    os.environ["HIGS_PX_RUNTIME"] = "2"                         # C0 V3 deploy: PX=2, block=128


# ============================================================================
# BUILD
# ============================================================================
_BUILD_SNIPPET = (
    "import os, sys, time, importlib.util\n"
    "work = os.environ['P3H_WORK']\n"
    "bf = os.path.join(work, 'experimental/render/kernels/cuda/build.py')\n"
    "p3 = int(os.environ['P3H_MODE'])\n"
    "spec = importlib.util.spec_from_file_location('p3h_build_%d' % p3, bf)\n"
    "m = importlib.util.module_from_spec(spec)\n"
    "sys.modules[spec.name] = m\n"
    "spec.loader.exec_module(m)\n"
    "t0 = time.time()\n"
    "mod = m.build_and_load_experimental_gaussian_render_inference_scene()\n"
    "dt = time.time() - t0\n"
    "print('SO_PATH', mod.__file__)\n"
    "print('BUILD_SECONDS', '%.3f' % dt)\n"
)


def _build_one(variant):
    n = VARIANTS[variant]["P3"]
    cache_v = os.path.join(CACHE_BASE, variant)
    os.makedirs(cache_v, exist_ok=True)
    env = dict(os.environ)
    env["PATH"] = ENV_BIN + ":" + env.get("PATH", "")
    env["CUDA_HOME"] = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env"
    env["TORCH_EXTENSIONS_DIR"] = cache_v
    env["NVCC_FLAGS"] = "-DP3_SINK_MODE=%d" % n
    env["P3H_WORK"] = WORK
    env["P3H_MODE"] = str(n)
    snippet = _BUILD_SNIPPET
    py = os.path.join(ENV_BIN, "python")
    t0 = time.time()
    proc = subprocess.run(
        [py, "-c", snippet], env=env, capture_output=True, text=True,
        cwd=WORK)
    dt = time.time() - t0
    out = proc.stdout
    so_path = None
    for line in out.splitlines():
        if line.startswith("SO_PATH"):
            so_path = line.split(" ", 1)[1].strip()
    if proc.returncode != 0 or so_path is None:
        raise RuntimeError(
            "BUILD FAILED for %s (mode=%s)\nSTDOUT:\n%s\nSTDERR:\n%s"
            % (variant, VARIANTS[variant]["mode"], out, proc.stderr[-8000:]))
    meta = {
        "variant": variant, "mode": VARIANTS[variant]["mode"],
        "P3_SINK_MODE": n, "so_path": so_path,
        "wall_seconds": dt, "build_time": time.strftime("%Y%m%dT%H%M%S"),
    }
    Path(build_meta_path(variant)).write_text(json.dumps(meta, indent=2))
    print("built %-3s mode=%-16s P3=%d wall=%.2fs -> %s" % (variant, VARIANTS[variant]["mode"], n, dt, so_path))
    return meta


def cmd_build(args):
    os.makedirs(REMOTE_ART, exist_ok=True)
    targets = list(VARIANTS) if args.all else args.variant
    if not targets:
        raise SystemExit("build: pass --variant Vn or --all")
    for v in targets:
        if v not in VARIANTS:
            raise SystemExit("unknown variant %r" % v)
        _build_one(v)
    print("\nVerify distinct .so below; list cache dirs:")
    for v in targets:
        p = so_path_for(v)
        print("  %s -> %s (%s bytes)" % (v, p, os.path.getsize(p) if os.path.exists(p) else "MISSING"))


# ============================================================================
# HASH
# ============================================================================
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout
        return out.strip().splitlines()[0]
    except Exception:
        return "unknown"


def cmd_hash(args):
    os.makedirs(REMOTE_ART, exist_ok=True)
    src = os.path.join(WORK, SINK_REL)
    src_sha = _sha256(src)
    gpu = _gpu_name()
    torch_v = torch.__version__ if torch is not None else "n/a"
    nvcc_v = "n/a"
    try:
        out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        nvcc_v = (out.stdout or out.stderr).strip().splitlines()[-1]
    except Exception:
        pass
    binaries = {}
    so_sizes = {}
    for v, info in VARIANTS.items():
        so = so_path_for(v)
        rec = {
            "variant": v, "mode": info["mode"], "P3_SINK_MODE": info["P3"],
            "so_path": so,
            "so_sha256": _sha256(so) if os.path.exists(so) else "MISSING",
            "source_cu_sha256": src_sha,
            "source_cu_path": src,
            "gpu": gpu, "nvcc": nvcc_v, "torch": torch_v,
            "exists": os.path.exists(so),
            "size_bytes": os.path.getsize(so) if os.path.exists(so) else -1,
        }
        if os.path.exists(so):
            so_sizes[v] = os.path.getsize(so)
        meta_f = build_meta_path(v)
        meta = json.loads(Path(meta_f).read_text()) if os.path.exists(meta_f) else {}
        rec["wall_build_seconds"] = meta.get("wall_seconds", None)
        binaries[v] = rec
    write(REMOTE_ART, "binary_hashes.json", binaries)

    provenance = {
        "task": "higs_p3_H_accum_ceiling (build+harness phase)",
        "run_ts": time.strftime("%Y%m%dT%H%M%S"),
        "gpu": gpu, "torch": torch_v, "nvcc": nvcc_v, "cuda_home_env": ENV_BIN,
        "torch_extensions_base_cache": CACHE_BASE,
        "extension_name": EXT_NAME,
        "baseline_identity": {
            "frozen_C0_V3_config": {
                "adjoint": "HIGS_BWD_SCALAR_ADJOINT=scalar_adjoint",
                "H8_moment_reduction": "HIGS_BWD_H8_MR=1",
                "px_runtime": "HIGS_PX_RUNTIME=2 (PX=2, block=128)",
            },
            "fixtures": {"scenes": ["room", "bicycle", "garden"], "max_long_side": 2048, "cam": 0},
        },
        "diagnostic_source": {
            "path": os.path.join(WORK, SINK_REL), "sha256": src_sha,
            "note": "P3_SINK_MODE-guarded diagnostic copy",
        },
        "frozen_production_source": {
            "path": os.path.join(FROZEN, "gsplat", SINK_REL),
            "sha256_prefix": FROZEN_SHA_PREFIX,
            "note": "frozen h8-mr production source; MUST remain untouched",
        },
        "so_sha256_by_variant": {v: binaries[v]["so_sha256"] for v in binaries},
        "so_size_bytes_by_variant": so_sizes,
    }
    write(REMOTE_ART, "provenance.json", provenance)
    print("hash done ->", REMOTE_ART)


# ============================================================================
# EMIT-STATIC
# ============================================================================
def cmd_emit_static(args):
    os.makedirs(REMOTE_ART, exist_ok=True)
    part = {
        "A": {
            "label": "pixel-local reverse compositing recurrence",
            "classification": "ORDER_DEPENDENT",
            "har_targetable": False,
            "note": "must preserve the reverse alpha/T recurrence across the pixel's Gaussian list; not a global reduction",
        },
        "B": {
            "label": "color adjoint accumulation (v_colors atomicAdd)",
            "classification": "ASSOCIATIVE_AFTER_PIXEL_RECURRENCE",
            "har_targetable": True,
        },
        "C": {
            "label": "opacity adjoint accumulation (v_opacities atomicAdd)",
            "classification": "ASSOCIATIVE_AFTER_PIXEL_RECURRENCE",
            "har_targetable": True,
        },
        "D": {
            "label": "H8 geometry-moment accumulation (v_conics 3x + v_means2d 2x atomicAdd)",
            "classification": "ASSOCIATIVE_AFTER_PIXEL_RECURRENCE",
            "har_targetable": True,
        },
        "E": {
            "label": "project/master Gaussian VJP (projection bwd: v_means/v_quats/v_scales gpuAtomicAdd)",
            "classification": "MANDATORY_GLOBAL_OUTPUT",
            "har_targetable": False,
        },
        "F": {
            "label": "SH VJP (v_coeffs gpuAtomicAdd) + means2d->means reduction",
            "classification": "MANDATORY_GLOBAL_OUTPUT",
            "har_targetable": False,
        },
        "G": {
            "label": "densification-facing updates",
            "classification": "MANDATORY_GLOBAL_OUTPUT",
            "har_targetable": False,
        },
        "H": {
            "label": "other (background kernel atomicAdd, reduce master)",
            "classification": "MANDATORY_GLOBAL_OUTPUT",
            "har_targetable": False,
        },
    }
    part["primary_har_targets"] = ["B", "C", "D"]
    part["note"] = (
        "V1 sinks B+C+D (appearance + geometry scatter block) to a per-warp "
        "uncontended scratch slot while preserving ALL local gradient "
        "arithmetic, isolating the global atomic-reduction / accumulation "
        "cost of the pixel-blend backward. E/F/G/H stay real in every variant."
    )
    write(REMOTE_ART, "backward_semantic_partition.json", part)

    acc = [
        {"id": "B", "site": "higs_blend_bwd_px_kernel", "line": 648,
         "code": "atomicAdd(v_rgb_ptr + k, v_rgb_local[k])", "loop": "CDIM",
         "field": "color", "atomic": "atomicAdd", "dtype": "float",
         "dest_tensor": "v_colors", "count_per_contributor": "1 per (pixel,contrib)",
         "har_eliminable": True},
        {"id": "C", "site": "higs_blend_bwd_px_kernel", "line": 650,
         "code": "atomicAdd(v_opacities + g, v_opacity_local)", "loop": "-",
         "field": "opacity", "atomic": "atomicAdd", "dtype": "float",
         "dest_tensor": "v_opacities", "count_per_contributor": "1 per (pixel,contrib)",
         "har_eliminable": True},
        {"id": "D1", "site": "higs_blend_bwd_px_kernel", "lines": [659, 660, 661],
         "code": "atomicAdd(v_conic_ptr + {0,1,2}, v_conic_local.{x,y,z})",
         "loop": "-", "field": "H8 conic", "atomic": "atomicAdd", "dtype": "float",
         "dest_tensor": "v_conics", "count_per_contributor": "3 per (pixel,contrib)",
         "har_eliminable": True},
        {"id": "D2", "site": "higs_blend_bwd_px_kernel", "lines": [662, 663],
         "code": "atomicAdd(v_xy_ptr + {0,1}, v_xy_local.{x,y})",
         "loop": "-", "field": "H8 xy", "atomic": "atomicAdd", "dtype": "float",
         "dest_tensor": "v_means2d", "count_per_contributor": "2 per (pixel,contrib)",
         "har_eliminable": True},
    ]
    # unguarded / not-in-scope atomic sites (left real in all variants):
    not_in_scope = [
        {"id": "X1", "site": "higs_background_bwd_kernel", "lines": [183, 436, 693],
         "code": "atomicAdd(v_backgrounds + k, ...)", "dest_tensor": "v_backgrounds",
         "har_eliminable": False, "note": "background kernel, NOT a V1 target"},
        {"id": "X2", "site": "higs_blend_bwd_kernel (non-px, unguarded)", "lines": [291, 295, 296, 297, 300, 301, 303],
         "code": "real atomicAdd, NOT guarded by P3_SINK_MODE", "dest_tensor": "v_colors/v_conics/v_means2d/v_opacities",
         "har_eliminable": False, "note": "non-PX blend kernel is not P3-guarded; stays real in all variants"},
        {"id": "X3", "site": "higs_projection_bwd_kernel", "lines": [891, 892, 893, 895, 896, 897, 898, 900, 901, 902],
         "code": "gpuAtomicAdd on v_means/v_quats/v_scales", "dest_tensor": "v_means, v_quats, v_scales",
         "har_eliminable": False, "note": "projection VJP, NOT in scope for V1"},
        {"id": "X4", "site": "higs_sh_vjp_grid_kernel", "lines": [922, 933, 934, 935, 967, 968, 969, 970, 971, 1031, 1032, 1033, 1034, 1035, 1036, 1037],
         "code": "gpuAtomicAdd(&v_coeffs[...], ...)", "dest_tensor": "v_coeffs",
         "har_eliminable": False, "note": "SH VJP, NOT in scope for V1"},
        {"id": "X5", "site": "higs_sh_vjp_grid_kernel", "lines": [1220, 1221, 1222, 1226, 1227, 1228],
         "code": "atomicAdd(v_means + m_id*3 + {0,1,2}, v_dir...)", "dest_tensor": "v_means",
         "har_eliminable": False, "note": "means2d->means global reduction, NOT in scope for V1"},
    ]
    sites_doc = {
        "blend_px_kernel": "higs_blend_bwd_px_kernel", "kernel_line": 318,
        "sites_in_scope_for_V1": acc,
        "sites_not_in_scope": not_in_scope,
        "primary_sink_targets": ["B", "C", "D"],
        "note": "Line numbers are read fresh from the P3-diag patched file "
                "(sha f01b624b...). All four in-scope atomicAdd sites are "
                "HAR-eliminable (associative global reduction). The unguarded "
                "non-PX blend kernel + background + projection/SH VJPs are NOT "
                "P3-guarded and stay real in every variant.",
    }
    write(REMOTE_ART, "accumulation_sites.json", sites_doc)

    vd = {}
    for v, info in VARIANTS.items():
        vd[v] = {
            "name": info["name"], "P3_SINK_MODE": info["P3"],
            "sunk": info["sunk"], "real": info["real"],
            "source_semantics": _variant_semantics(v),
        }
    write(REMOTE_ART, "variant_definitions.json", vd)
    print("emit-static done ->", REMOTE_ART)


def _variant_semantics(v):
    return {
        "V0": "Frozen C0 V3: every per-Gaussian gradient scatters to the true "
              "global tensor via atomicAdd (B color, C opacity, D H8 moments). "
              "Backward accumulation ceiling reference.",
        "V1": "B+C+D gradient scatter block replaced by an uncontended per-warp "
              "scratch sink; all local arithmetic (q eval, alpha derivative, T "
              "recurrence, warp reduction, index mapping) preserved.",
        "V2": "Only D (H8 conic 3x + xy 2x) scatter atoms are sunk; B color and "
              "C opacity remain real atomicAdd to v_colors/v_opacities.",
        "V3": "Only B+C appearance scatter atoms are sunk; D geometry remains "
              "real atomicAdd to v_conics/v_means2d.",
        "V4": "Full values written to the scratch slot via plain non-atomic "
              "stores (no atomic RMW/contention anywhere in the scatter)",
    }[v]


# ============================================================================
# HARNESS (fixture + backward launch)  -- shared by smoke & timing
# ============================================================================
def _bootstrap_gsplat():
    """Make ``import gsplat`` resolve to the DIAGNOSTIC worktree package root."""
    boot = "/tmp/p3h_boot"
    os.makedirs(boot, exist_ok=True)
    link = os.path.join(boot, "gsplat")
    if not os.path.exists(link):
        os.symlink(WORK, link)
    if boot not in sys.path:
        sys.path.insert(0, boot)


def load_backend(variant, so_path):
    # The .so is built with TORCH_EXTENSION_NAME=experimental_gaussian_render_inference_scene_cuda,
    # so the importable module name MUST be that exact string (PyInit_<name> lookup).
    # Use a per-variant sys.modules key to avoid clobbering across variants while
    # keeping the loader-name correct for the PyInit symbol.
    spec = importlib.util.spec_from_file_location(
        EXT_NAME, so_path)
    ext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext)
    sys.modules["gsplat_p3h_%s" % variant] = ext
    # CRITICAL: register this variant ALSO as the worktree's experimental-kernels
    # csrc module so that importing gsplat.experimental...gaussian_inference does
    # NOT try to pull the frozen environment's csrc.so (which would register the
    # same torch TORCH_LIBRARY namespace "experimental" and crash the process).
    # With this sys.modules entry, `from ..kernels import csrc` returns our module.
    sys.modules["gsplat.experimental.render.kernels.csrc"] = ext
    return ext


def _load_ply_scene(ply_path, device):
    from plyfile import PlyData
    v = PlyData.read(ply_path)["vertex"]
    means = torch.tensor(np.column_stack([v["x"], v["y"], v["z"]]), device=device, dtype=torch.float32)
    rot_cols = [v["rot_%d" % i] for i in range(4)]
    quats = torch.tensor(np.column_stack(rot_cols), device=device, dtype=torch.float32)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale_cols = [v["scale_%d" % i] for i in range(3)]
    scales = torch.exp(torch.tensor(np.column_stack(scale_cols), device=device, dtype=torch.float32))
    opacities = torch.sigmoid(torch.tensor(v["opacity"], device=device, dtype=torch.float32))
    K_SH = (SH_DEGREE + 1) ** 2
    sh = torch.zeros((len(v), K_SH, 3), device=device, dtype=torch.float32)
    dc_cols = [v["f_dc_%d" % i] for i in range(3)]
    sh[:, 0] = torch.tensor(np.column_stack(dc_cols), device=device, dtype=torch.float32)
    rest = torch.stack([torch.tensor(v["f_rest_%d" % i], device=device, dtype=torch.float32) for i in range(45)], 1)
    sh[:, 1:] = rest.reshape(len(v), 3, 15).permute(0, 2, 1)
    return means, quats, scales, opacities, sh


def _load_cams(cams_path, width, height, device):
    cams = json.loads(Path(cams_path).read_text())
    c = cams[0]
    native_w, native_h = int(c["width"]), int(c["height"])
    R = np.asarray(c["rotation"], dtype=np.float32).T
    p = np.asarray(c["position"], dtype=np.float32)
    vm = np.eye(4, dtype=np.float32)
    vm[:3, :3] = R
    vm[:3, 3] = -R @ p
    K = np.array([[float(c["fx"]) * width / native_w, 0, (width - 1) / 2],
                  [0, float(c["fy"]) * width / native_w, (height - 1) / 2],
                  [0, 0, 1]], dtype=np.float32)
    return torch.tensor(vm, device=device)[None, None], torch.tensor(K, device=device)[None, None]


def build_fixture(scene, max_long_side, device, ext, show_run_id=True):
    """Forward-capture the fixture tensors the native backward consumes.

    Mirrors gsplat.experimental...gaussian_inference._native_forward_capture.
    Returns a fixture dict with the captured tensors + master grad accumulators.
    """
    from gsplat.experimental.render.functional import gaussian_inference as gi
    gi._HigsAutogradFunction  # ensure module imported
    from gsplat.cuda._wrapper import fully_fused_projection, isect_offset_encode

    cfg = SCENE_CONFIGS[scene]
    s = min(1.0, max_long_side / max(cfg["native_w"], cfg["native_h"]))
    width, height = round(cfg["native_w"] * s), round(cfg["native_h"] * s)
    tw, th = math.ceil(width / TILE_SIZE), math.ceil(height / TILE_SIZE)

    means, quats, scales, opacities, colors = _load_ply_scene(cfg["ply"], device)
    vm, K = _load_cams(cfg["cams"], width, height, device)
    vm, K = vm[:, [0]], K[:, [0]]

    with torch.no_grad():
        ids, _m, _r = gi._cull_gaussians_batched(
            means, quats, scales, vm, K, width, height,
            eps2d=EPS2D, near_plane=NEAR, far_plane=FAR, radius_clip=RADIUS_CLIP,
            camera_model=CAMERA_MODEL)
        v_means, v_quats, v_scales, v_opacities, v_colors = gi._gather_visible_native(
            means, quats, scales, opacities, colors, ids)
        N_vis = len(v_means)
        # capture via the wrapper's own single-pass forward
        ctx = types.SimpleNamespace()
        backgrounds_b = torch.zeros(3, device=device, dtype=torch.float32)
        _set_runtime_env()
        render_colors, render_alphas, captured = gi._HigsAutogradFunction._native_forward_capture(
            ctx,
            v_means.unsqueeze(0), v_quats.unsqueeze(0), v_scales.unsqueeze(0),
            v_opacities.unsqueeze(0),
            v_colors,                                    # [N, K, D]
            vm, K, width, height, SH_DEGREE, TILE_SIZE,
            NEAR, FAR, RADIUS_CLIP, EPS2D,
            backgrounds_b, CAMERA_MODEL, RENDER_MODE,
            1.0, "uniform", None, 1)
        D = captured[2].shape[-1]  # nch (3 for RGB)

    # master gradient accumulators (full master sizes, written via visible_ids)
    grad_means = torch.zeros_like(means)
    grad_quats = torch.zeros_like(quats)
    grad_scales = torch.zeros_like(scales)
    grad_opacities = torch.zeros_like(opacities)
    grad_colors = torch.zeros_like(colors)
    bg_kernel = torch.zeros((1, 3), device=device, dtype=torch.float32)

    fixture = {
        "scene": scene, "width": width, "height": height, "tw": tw, "th": th,
        "n_vis": N_vis, "n_isects": int(captured[5].numel()),
        "means2d": captured[0], "conics": captured[1], "colors_eval": captured[2],
        "opacities_f": captured[3], "tile_offsets": captured[4],
        "flatten_ids": captured[5], "render_alphas": captured[6],
        "last_ids": captured[7], "radii": captured[8], "depth_acc": captured[9],
        "v_means": v_means, "v_quats": v_quats, "v_scales": v_scales,
        "v_opacities": v_opacities, "v_colors": v_colors,
        "viewmats": vm, "Ks": K, "bg_kernel": bg_kernel, "channel_count": D,
        "visible_ids": ids,            # [N_vis] master row per visible gaussian
        "grad_means": grad_means, "grad_quats": grad_quats,
        "grad_scales": grad_scales, "grad_opacities": grad_opacities,
        "grad_colors": grad_colors,
        "render_output": render_colors, "render_alpha_output": render_alphas,
    }
    return fixture


def _backward_kwargs(fx):
    I = 1
    H, W = fx["height"], fx["width"]
    return dict(
        means2d=fx["means2d"],
        conics=fx["conics"],
        colors_eval=fx["colors_eval"],
        opacities=fx["opacities_f"],
        backgrounds=fx["bg_kernel"],
        tile_offsets=fx["tile_offsets"],
        flatten_ids=fx["flatten_ids"],
        active_tiles=None,
        render_alphas=fx["render_alphas"],
        last_ids=fx["last_ids"],
        means=fx["v_means"],
        quats=fx["v_quats"],
        scales=fx["v_scales"],
        radii=fx["radii"],
        viewmats=fx["viewmats"],
        Ks=fx["Ks"],
        width=W, height=H, tile_size=TILE_SIZE, eps2d=EPS2D, camera_model=0,
        v_render_colors=torch.ones((I, H, W, fx["channel_count"]), device=fx["v_means"].device),
        v_render_alphas=torch.ones((I, H, W), device=fx["v_means"].device),
        sh_coeffs=fx["v_colors"],
        sh_degree=SH_DEGREE,
        visible_ids=fx["visible_ids"],
        grad_means=fx["grad_means"],
        grad_quats=fx["grad_quats"],
        grad_scales=fx["grad_scales"],
        grad_opacities=fx["grad_opacities"],
        grad_colors=fx["grad_colors"],
    )


def _forward_once(fx):
    """Run the differentiable native forward once (no autograd graph kept) and
    return a freshly captured fixture for a forward+backward measurement."""
    # Re-run capture as the 'forward' (rebuilds isects / hits the rasterize
    # forward kernel identically to the true forward pass).
    from gsplat.experimental.render.functional import gaussian_inference as gi
    ctx = types.SimpleNamespace()
    _set_runtime_env()
    render_colors, render_alphas, captured = gi._HigsAutogradFunction._native_forward_capture(
        ctx, fx["v_means"].unsqueeze(0), fx["v_quats"].unsqueeze(0),
        fx["v_scales"].unsqueeze(0), fx["v_opacities"].unsqueeze(0),
        fx["v_colors"], fx["viewmats"], fx["Ks"], fx["width"], fx["height"],
        SH_DEGREE, TILE_SIZE, NEAR, FAR, RADIUS_CLIP, EPS2D,
        fx["bg_kernel"].reshape(-1), CAMERA_MODEL, RENDER_MODE,
        1.0, "uniform", None, 1)
    fx2 = dict(fx)
    fy = {
        "means2d": captured[0], "conics": captured[1], "colors_eval": captured[2],
        "opacities_f": captured[3], "tile_offsets": captured[4],
        "flatten_ids": captured[5], "render_alphas": captured[6],
        "last_ids": captured[7], "radii": captured[8], "depth_acc": captured[9],
    }
    fx2.update(fy)
    return fx2


def _run_backward(ext, fx):
    kwargs = _backward_kwargs(fx)
    out = ext.higs_rasterize_backward(**kwargs)
    return kwargs, out


def _assert_finite(*tensors):
    for name, t in tensors:
        if not torch.isfinite(t).all().item():
            return "non-finite in %s" % name
    return None


def smoke_variant(ext, variant, device):
    _set_runtime_env()
    fx = build_fixture("room", MAX_LONG_SIDE, device, ext)
    # warmup
    torch.cuda.synchronize()
    _run_backward(ext, fx)
    torch.cuda.synchronize()
    # timed backward-only
    s0 = torch.cuda.Event(enable_timing=True); e0 = torch.cuda.Event(enable_timing=True)
    s0.record()
    kwargs, out = _run_backward(ext, fx)
    e0.record(); torch.cuda.synchronize()
    bwd_ms = s0.elapsed_time(e0)
    # timed forward (capture) + backward
    s1 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    s1.record()
    fx2 = _forward_once(fx)
    _run_backward(ext, fx2)
    e1.record(); torch.cuda.synchronize()
    fb_ms = s1.elapsed_time(e1)

    grad_accs = [g for g in (fx["grad_means"], fx["grad_quats"], fx["grad_scales"],
                             fx["grad_opacities"], fx["grad_colors"])]
    err = None
    for g in grad_accs:
        err = _assert_finite(("grad", g))
        if err:
            break
    if err is None:
        # all 7 outputs finite
        for o in out:
            err = _assert_finite(("out", o))
            if err:
                break
    return {
        "variant": variant, "mode": VARIANTS[variant]["mode"],
        "pass": err is None, "error": err,
        "smoke_bwd_ms": round(bwd_ms, 4), "smoke_fb_ms": round(fb_ms, 4),
        "n_vis": fx["n_vis"], "n_isects": fx["n_isects"],
        "WxH": "%dx%d" % (fx["width"], fx["height"]),
    }


def _smoke_one_sub(variant):
    """Run the smoke for a single variant in an isolated subprocess (each .so
    registers the torch TORCH_LIBRARY namespace 'experimental', so multiple
    variant .so cannot coexist in one process). Returns the result dict."""
    import json
    py = os.path.join(ENV_BIN, "python")
    script = (
        "import sys, json\n"
        "sys.path.insert(0, '/tmp')\n"  # ensure p3_h_accum_ceiling.py importable
        "import p3_h_accum_ceiling as M\n"
        "M._bootstrap_gsplat()\n"
        "import torch\n"
        "v = %r\n" % variant +
        "dev = torch.device('cuda')\n"
        "so = M.so_path_for(v)\n"
        "ext = M.load_backend(v, so)\n"
        "try:\n"
        "    r = M.smoke_variant(ext, v, dev)\n"
        "    r['tag'] = M.GPU_TAG\n"
        "except Exception as e:\n"
        "    import traceback; traceback.print_exc()\n"
        "    r = {'variant': v, 'pass': False, 'error': repr(e), 'tag': M.GPU_TAG}\n"
        "print('SMOKE_RESULT ' + json.dumps(r))\n"
    )
    env = dict(os.environ)
    env["PATH"] = ENV_BIN + ":" + env.get("PATH", "")
    env["CUDA_HOME"] = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env"
    proc = subprocess.run([py, "-c", script], env=env, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("SMOKE_RESULT "):
            return json.loads(line[len("SMOKE_RESULT "):])
    return {"variant": variant, "pass": False,
            "error": "no SMOKE_RESULT; rc=%s stderr=%s" % (proc.returncode, proc.stderr[-2000:])}


def cmd_smoke(args):
    os.makedirs(REMOTE_ART, exist_ok=True)
    results = []
    for v in list(VARIANTS):
        so = so_path_for(v)
        if not os.path.exists(so):
            results.append({"variant": v, "pass": False, "error": "so missing"})
            continue
        r = _smoke_one_sub(v)
        results.append(r)
        print("%s  PASS=%s  smoke_bwd=%sms  smoke_fb=%sms  WxH=%s" % (
            v, r.get("pass"), r.get("smoke_bwd_ms"), r.get("smoke_fb_ms"), r.get("WxH")), flush=True)
    write(REMOTE_ART, "smoke_results.json", {"tag": GPU_TAG, "results": results})
    npass = sum(1 for r in results if r.get("pass"))
    print("\nsmoke: %d/%d variants PASS (%s)" % (npass, len(results), GPU_TAG))


# ============================================================================
# TIMING (implemented; run later on an idle GPU)
# ============================================================================
def _time_rep(ext, fx, kind, samples_per_rep):
    """kind: 'backward' or 'fb'. Returns per-call avg ms over samples."""
    device = fx["v_means"].device
    if kind == "backward":
        kwargs, _ = _run_backward(ext, fx)
    # warmup
    for _ in range(3):
        if kind == "fb":
            _run_backward(ext, _forward_once(fx))
        else:
            _run_backward(ext, fx)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(samples_per_rep):
        if kind == "fb":
            fx = _forward_once(fx)
        _run_backward(ext, fx)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / samples_per_rep, None


def _write_rows_csv(path, rows):
    header = ["scene", "variant", "rep", "kind", "value_ms", "error"]
    Path(path).write_text(
        ",".join(header) + "\n" +
        "\n".join(",".join(str(r.get(k, "")) for k in header) for r in rows) + "\n")


def _timing_one_variant(v, scenes, warmup, reps, samples):
    """Run the full timing protocol for ONE variant in a dedicated subprocess.

    Each variant .so registers the torch TORCH_LIBRARY namespace
    'experimental'; a second variant .so crashes dlopen in the same process
    (verified empirically). Isolate one variant per process, load its backend
    once, run all scenes x reps x kinds, and write a per-variant rows CSV.
    """
    import torch
    dev = torch.device("cuda")
    ext = load_backend(v, so_path_for(v))
    rows = []
    for scene in scenes:
        fx = build_fixture(scene, MAX_LONG_SIDE, dev, ext)
        for rep_i in range(reps):
            for kind in ("backward", "fb"):
                for _ in range(warmup):
                    if kind == "fb":
                        _run_backward(ext, _forward_once(fx))
                    else:
                        _run_backward(ext, fx)
                torch.cuda.synchronize()
                ms, err = _time_rep(ext, fx, kind, samples)
                rows.append({"scene": scene, "variant": v, "rep": rep_i,
                             "kind": kind,
                             "value_ms": (round(ms, 5) if not err else None),
                             "error": err or ""})
    _write_rows_csv(os.path.join(REMOTE_ART, "timing_rows_%s.csv" % v), rows)
    print("VARIANT_DONE %s rows=%d" % (v, len(rows)))


def cmd_timing(args):
    os.makedirs(REMOTE_ART, exist_ok=True)
    warmup = args.warmup
    reps = args.reps
    samples = args.samples
    scenes = args.scenes.split(",") if args.scenes else list(SCENE_CONFIGS)
    variants = list(VARIANTS) if args.all else args.variant
    if not variants:
        raise SystemExit("timing: pass --variant Vn or --all")

    import subprocess as _sp
    py = os.path.join(ENV_BIN, "python")
    for v in variants:
        # One fresh python subprocess per variant (variant .so cannot coexist).
        script = (
            "import sys\n"
            "sys.path.insert(0, '/mnt/storage_pool/liaoyuanjun')\n"
            "import p3_h_run as M\n"
            "M._bootstrap_gsplat()\n"
            "import torch\n"  # must import torch BEFORE dlopen of the variant .so
            "M._timing_one_variant(%r, %r, %d, %d, %d)\n" % (v, scenes, warmup, reps, samples)
        )
        env = dict(os.environ)
        env["PATH"] = ENV_BIN + ":" + env.get("PATH", "")
        env["CUDA_HOME"] = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env"
        env["HIGS_BWD_SCALAR_ADJOINT"] = "scalar_adjoint"
        env["HIGS_BWD_H8_MR"] = "1"
        env["HIGS_PX_RUNTIME"] = "2"
        proc = _sp.run([py, "-c", script], env=env, capture_output=True, text=True)
        for line in (proc.stdout or "").splitlines():
            if line.startswith("VARIANT_DONE"):
                print(line, flush=True)
        if proc.returncode != 0:
            print("timing: variant %s subprocess rc=%s" % (v, proc.returncode), flush=True)
            print((proc.stdout or "")[-4000:], flush=True)
            print((proc.stderr or "")[-4000:], flush=True)

    # Aggregate per-variant rows into one table.
    import csv as _csv
    rows = []
    for v in variants:
        p = os.path.join(REMOTE_ART, "timing_rows_%s.csv" % v)
        if not os.path.exists(p):
            rows.append({"scene": "", "variant": v, "rep": "", "kind": "",
                         "value_ms": None, "error": "missing timing_rows csv"})
            continue
        with open(p) as f:
            for r in _csv.DictReader(f):
                rows.append({
                    "scene": r["scene"], "variant": r["variant"],
                    "rep": int(r["rep"]) if r["rep"] not in ("", "None") else "",
                    "kind": r["kind"],
                    "value_ms": (float(r["value_ms"])
                                 if r["value_ms"] not in ("", "None") else None),
                    "error": r["error"],
                })
    write(REMOTE_ART, "raw_timing.csv",
          (",").join(["scene", "variant", "rep", "kind", "value_ms", "error"]) + "\n" +
          "\n".join(",".join(str(rows[i].get(k, "")) for k in
                             ["scene", "variant", "rep", "kind", "value_ms", "error"])
                    for i in range(len(rows))))
    # summary per (scene, variant, kind) over reps
    import statistics as st
    summary = {"tag": GPU_TAG if args.gpu_shared else "idle_gpu",
               "warmup": warmup, "reps": reps, "samples_per_rep": samples}
    agg = {}
    for scene in scenes:
        for v in variants:
            for kind in ("backward", "fb"):
                vals = [r["value_ms"] for r in rows
                        if r["scene"] == scene and r["variant"] == v
                        and r["kind"] == kind and r.get("value_ms") is not None]
                doc = {}
                if vals:
                    doc = {"mean": round(st.mean(vals), 5), "median": round(st.median(vals), 5),
                           "std": round(st.pstdev(vals), 5),
                           "p10": round(float(np.percentile(vals, 10)), 5),
                           "p90": round(float(np.percentile(vals, 90)), 5), "n": len(vals)}
                agg.setdefault(scene, {})[v + "/" + kind] = doc
    summary["results"] = agg
    write(REMOTE_ART, "timing_summary.json", summary)
    print("timing done ->", REMOTE_ART)


# ============================================================================
# utils
# ============================================================================
def write(dirp, name, obj):
    os.makedirs(dirp, exist_ok=True)
    p = os.path.join(dirp, name)
    if isinstance(obj, str):
        Path(p).write_text(obj)
    else:
        Path(p).write_text(json.dumps(obj, indent=2, default=float))
    print("wrote %s" % p)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build")
    pb.add_argument("--variant", nargs="*", default=[])
    pb.add_argument("--all", action="store_true")
    pb.set_defaults(fn=cmd_build)

    ph = sub.add_parser("hash")
    ph.set_defaults(fn=cmd_hash)

    pe = sub.add_parser("emit-static")
    pe.set_defaults(fn=cmd_emit_static)

    ps = sub.add_parser("smoke")
    ps.set_defaults(fn=cmd_smoke)

    pt = sub.add_parser("timing")
    pt.add_argument("--variant", nargs="*", default=[])
    pt.add_argument("--all", action="store_true")
    pt.add_argument("--scenes", default=None)
    pt.add_argument("--warmup", type=int, default=20)
    pt.add_argument("--reps", type=int, default=5)
    pt.add_argument("--samples", type=int, default=100)
    pt.add_argument("--gpu-shared", action="store_true")
    pt.set_defaults(fn=cmd_timing)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
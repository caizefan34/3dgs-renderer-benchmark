#!/usr/bin/env python3
"""Build the true-accutile B1A extension for FINAL-30K under higs-13scene-env (torch 2.9.1+cu128).

The tree's own JIT loader (_backend.py) is torch-2.4-era (and has a latent
list-vs-string bug in its cuda flags), and its prebuilt gsplat/csrc.so targets
the torch 2.4.1 ABI. This builder replicates the tree's setup.py build EXACTLY
(same sources, include dirs, cxx/nvcc flags, -s link flag) via
torch.utils.cpp_extension.load into a separate cache directory, leaving the
frozen source tree read-only and the historical artifacts untouched.

Writes /mnt/storage_pool/liaoyuanjun/b1a_accutile_build_identity.json
"""
import glob
import hashlib
import json
import os
import sys

TREE = "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153"
CACHE = "/mnt/storage_pool/liaoyuanjun/gsplat_accutile_final30k_cache"
NAME = "gsplat_cuda_final30k"

import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

out_dir = os.path.join(CACHE, NAME)
os.makedirs(out_dir, exist_ok=True)

ext_dir = os.path.join(TREE, "gsplat", "cuda")
sources = (
    sorted(glob.glob(os.path.join(ext_dir, "csrc", "*.cu")))
    + sorted(glob.glob(os.path.join(ext_dir, "csrc", "*.cpp")))
    + [os.path.join(ext_dir, "ext.cpp")]
)
assert sources, "no sources found"
print(f"{len(sources)} sources")

glm_path = os.path.join(ext_dir, "csrc", "third_party", "glm")
include_dirs = [glm_path, os.path.join(ext_dir, "include")]

# EXACT setup.py flags of the frozen tree (python setup.py build era):
extra_cflags = ["-O3", "-Wno-sign-compare", "-DAT_PARALLEL_OPENMP", "-fopenmp"]
extra_cuda_cflags = [
    "-O3", "--use_fast_math", "-std=c++17", "--extended-lambda",
    "--expt-relaxed-constexpr", "-diag-suppress", "20012,186",
]
extra_ldflags = ["-s"]

os.makedirs(CACHE, exist_ok=True)
mod = load(
    name=NAME,
    sources=sources,
    extra_cflags=extra_cflags,
    extra_cuda_cflags=extra_cuda_cflags,
    extra_ldflags=extra_ldflags,
    extra_include_paths=include_dirs,
    build_directory=out_dir,
    verbose=False,
)
so_path = mod.__file__
print(f"built: {so_path}")

h = hashlib.sha256()
with open(so_path, "rb") as f:
    for c in iter(lambda: f.read(1 << 20), b""):
        h.update(c)
sha = h.hexdigest()

# sanity: required symbol present (bwd is the only entry point the D2 gate needs)
assert hasattr(mod, "rasterize_to_pixels_3dgs_bwd"), "missing rasterize_to_pixels_3dgs_bwd"

src_manifest = {}
for s in sources:
    hh = hashlib.sha256()
    with open(s, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            hh.update(c)
    src_manifest[os.path.relpath(s, TREE)] = hh.hexdigest()

ident = {
    "accutile_tree": TREE,
    "extension_name": NAME,
    "extension_path": so_path,
    "extension_sha256": sha,
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "python": sys.version.split()[0],
    "arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST", "(auto)"),
    "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "(all)"),
    "build_flags": {"cxx": extra_cflags, "nvcc": extra_cuda_cflags, "ld": extra_ldflags},
    "n_sources": len(sources),
    "source_manifest": src_manifest,
    "setup_py_replication": "flags/source-set/include-dirs replicate the tree's setup.py (historical binary build path); tree's own _backend.py JIT path not used (torch-2.4-era signature + latent list-arg bug)",
    "purpose": "B1A baseline extension for FINAL-30K matched env (torch 2.9.1+cu128)",
}
out = "/mnt/storage_pool/liaoyuanjun/b1a_accutile_build_identity.json"
with open(out, "w") as f:
    json.dump(ident, f, indent=2)
print(json.dumps({k: v for k, v in ident.items() if k != "source_manifest"}, indent=2))
print(f"WROTE {out}")

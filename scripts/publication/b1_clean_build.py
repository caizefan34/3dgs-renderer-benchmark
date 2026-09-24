#!/usr/bin/env python3
"""Build the B1_CLEAN_GSPLAT extension: pristine upstream gsplat v1.5.3
(937e29912570c372bed6747a5c9bf85fed877bae) under higs-13scene-env
(torch 2.9.1+cu128), replicating the b1a_build.py build pattern with a
distinct extension name. Writes an identity manifest with source hashes.
"""
import glob
import hashlib
import json
import os
import sys

TREE = "/mnt/storage_pool/liaoyuanjun/gsplat-b1-clean-v153"
CACHE = "/mnt/storage_pool/liaoyuanjun/gsplat_b1clean_pub_cache"
NAME = "gsplat_cuda_b1clean"

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

# Same flag set as the frozen tree's setup.py (identical to b1a_build.py).
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

# sanity: the 3DGS fwd/bwd entry points exist
for sym in ("rasterize_to_pixels_3dgs_fwd", "rasterize_to_pixels_3dgs_bwd"):
    assert hasattr(mod, sym), f"missing {sym}"

src_manifest = {}
for s in sources:
    hh = hashlib.sha256()
    with open(s, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            hh.update(c)
    src_manifest[os.path.relpath(s, TREE)] = hh.hexdigest()

ident = {
    "variant": "B1_CLEAN_GSPLAT",
    "tree": TREE,
    "base_commit": "937e29912570c372bed6747a5c9bf85fed877bae",
    "base_tag": "v1.5.3",
    "pristine_verification": "accutile grep hits=0 after git checkout revert of the 9 patched files; git status clean (submodules ignored)",
    "extension_name": NAME,
    "extension_path": so_path,
    "extension_sha256": sha,
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "python": sys.version.split()[0],
    "arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST", "(auto)"),
    "build_flags": {"cxx": extra_cflags, "nvcc": extra_cuda_cflags, "ld": extra_ldflags},
    "n_sources": len(sources),
    "source_manifest": src_manifest,
    "role": "P1 publication baseline: clean modern gsplat, no C0 optimizations, no accutile",
}
out = "/mnt/storage_pool/liaoyuanjun/b1_clean_build_identity.json"
with open(out, "w") as f:
    json.dump(ident, f, indent=2)
print(json.dumps({k: v for k, v in ident.items() if k != "source_manifest"}, indent=2))
print(f"WROTE {out}")

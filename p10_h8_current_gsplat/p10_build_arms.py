#!/usr/bin/env python3
"""P10 build: pristine + H8-MR patched gsplat-main extensions.

Both arms are built from gsplat main @ 512d366b67073d77ca099ede742683c165dfc23b
with flags taken verbatim from the tree's own gsplat/cuda/build.py (release
configuration); only the torch extension NAME is overridden so the two build
products can coexist in one environment. Nothing else differs.

  --arm pristine  tree_pristine (upstream blobs)
  --arm h8        tree_h8       (upstream + P10_H8_CURRENT_GSPLAT_PATCH.diff)
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

ROOT = "/root/epic05_p10"
TREES = {"pristine": f"{ROOT}/tree_pristine", "h8": f"{ROOT}/tree_h8",
         "pristine_ctl": f"{ROOT}/tree_pristine_ctl"}
NAMES = {"pristine": "gsplat_cuda_p10_pristine", "h8": "gsplat_cuda_p10_h8",
         "pristine_ctl": "gsplat_cuda_p10_pristine_ctl"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(TREES), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_HOME", "/root/miniforge3/envs/gsplat")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    os.environ.setdefault("MAX_JOBS", "32")
    os.environ.pop("DEBUG", None)
    os.environ.pop("NVCC_FLAGS", None)

    tree = TREES[args.arm]
    name = NAMES[args.arm]
    build_dir = f"{ROOT}/build/{name}"
    os.makedirs(build_dir, exist_ok=True)

    import torch
    from torch.utils import cpp_extension as jit

    build_py = os.path.join(tree, "gsplat", "cuda", "build.py")
    spec = importlib.util.spec_from_file_location("gsplat_build_p10", build_py)
    gm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gm)
    p = gm.get_build_parameters()

    print(f"[{args.arm}] name={name} sources={len(p.sources)} build_dir={build_dir}")
    jit.load(
        name=name,
        sources=p.sources,
        extra_cflags=p.extra_cflags,
        extra_cuda_cflags=p.extra_cuda_cflags,
        extra_ldflags=p.extra_ldflags,
        extra_include_paths=p.extra_include_paths,
        build_directory=build_dir,
        verbose=True,
    )

    so = os.path.join(build_dir, name + ".so")
    assert os.path.exists(so), f"expected .so missing: {so}"

    ninja = os.path.join(build_dir, "build.ninja")
    nvcc_line = ""
    if os.path.exists(ninja):
        with open(ninja) as f:
            for line in f:
                if "nvcc" in line and ".cu.o" in line:
                    nvcc_line = line.strip()
                    break

    ident = {
        "arm": args.arm,
        "extension_name": name,
        "tree": tree,
        "so_path": so,
        "so_sha256": sha256_file(so),
        "so_size": os.path.getsize(so),
        "upstream_sha": "512d366b67073d77ca099ede742683c165dfc23b",
        "n_sources": len(p.sources),
        "source_manifest": {
            os.path.relpath(s, tree): sha256_file(s) for s in p.sources
        },
        "build_flags": {
            "cxx": p.extra_cflags,
            "nvcc": p.extra_cuda_cflags,
            "ld": p.extra_ldflags,
            "include": p.extra_include_paths,
        },
        "build_ninja_sha256": sha256_file(ninja) if os.path.exists(ninja) else None,
        "nvcc_command_example": nvcc_line,
        "env": {
            "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST"),
            "MAX_JOBS": os.environ.get("MAX_JOBS"),
            "CUDA_HOME": os.environ.get("CUDA_HOME"),
            "DEBUG": os.environ.get("DEBUG", "0"),
            "FAST_MATH": os.environ.get("FAST_MATH", "1"),
            "WITH_SYMBOLS": os.environ.get("WITH_SYMBOLS", "0"),
        },
        "versions": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "python": sys.version.split()[0],
            "nvcc": subprocess.run(
                [os.path.join(jit.CUDA_HOME, "bin", "nvcc"), "--version"],
                capture_output=True, text=True,
            ).stdout.strip().splitlines()[-1],
            "gpu": (None if os.environ.get("P10_SKIP_GPU_QUERY") == "1"
                    else torch.cuda.get_device_name(0)),
            "gpu_uuid": (None if os.environ.get("P10_SKIP_GPU_QUERY") == "1"
                         else subprocess.run(
                             ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                             capture_output=True, text=True,
                         ).stdout.strip().splitlines()[0]),
        },
    }
    with open(args.out, "w") as f:
        json.dump(ident, f, indent=2)
    print(f"[{args.arm}] OK sha256={ident['so_sha256'][:16]} -> {args.out}")


if __name__ == "__main__":
    main()

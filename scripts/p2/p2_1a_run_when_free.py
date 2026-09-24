#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2-1A — next-GPU execution package
==================================
Runs the compile-only FP32 native-hierarchy forward prototype on a GPU once an
A100 is actually free. Strictly ordered gates:

    0. GPU idle verification          (memset first-boot only, no kernels)
    1. load prototype (.so)
    2. structural equivalence
    3. ordering equivalence
    4. RGB / alpha correctness
    5. resource runtime confirmation  (torch.cuda usage / nvml)
    6. ONLY THEN timing

The script ABORTS (sys.exit) before timing if any correctness/structural gate
fails, as mandated by P2-1A. It is intentionally not invoked now: every A100 is
occupied.

DO NOT run this on an occupied GPU. On entry it enumerates active processes on
every visible CUDA device and refuses / aborts with a clear message unless all
are idle.

Invocation (as soon as a free A100 exists):
    CUDA_VISIBLE_DEVICES=<free_gpu> \
    TORCH_EXTENSIONS_DIR=/mnt/storage_pool/liaoyuanjun/higs_p2_1a_cache \
    python p2_1a_run_when_free.py --proto <path/to/gsplat_cuda_p2_1a.so>

No performance claims and no unexecuted-kernel correctness claims are made here;
this harness is what actually produces them.
"""

import argparse
import os
import sys
import time


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------

def _fatal(msg: str) -> None:
    print(f"[P2-1A] ABORT: {msg}", flush=True)
    sys.exit(2)


def _step(n: int, title: str) -> None:
    print(f"\n[P2-1A] ---- step {n}: {title} ----", flush=True)


def check_gpus_idle() -> None:
    """Step 0: refuse to proceed if any visible GPU has an active process."""
    _step(0, "GPU idle verification")
    try:
        import torch
        import pynvml  # available in the frozen env
    except ImportError as e:  # pragma: no cover - env-specific
        _fatal(f"missing dependency for gate 0: {e}")
    n = torch.cuda.device_count()
    if n < 1:
        _fatal("no CUDA device visible; refusing to run anywhere.")
    try:
        pynvml.nvmlInit()
        busy = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            # By MIG or whole-device: any process == busy.
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            if procs:
                busy.append((i, name.decode() if isinstance(name, bytes) else name, len(procs)))
        if busy:
            _fatal(
                "active GPU processes found; refusing to run P2-1A timing on occupied "
                f"hardware. Details: {busy}"
            )
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    print(f"[P2-1A] {n} device(s) present, no active processes. Proceeding.", flush=True)


# ---------------------------------------------------------------------------
# Step 1: load prototype
# ---------------------------------------------------------------------------

def load_prototype(proto_path: str):
    _step(1, "load prototype")
    if not proto_path or not os.path.isfile(proto_path):
        _fatal(f"prototype .so not found: {proto_path}")
    # Verify it is the expected ELF artifact (not a sandbox stub) by size floor.
    if os.path.getsize(proto_path) < 100_000:
        _fatal("prototype .so implausibly small; refusing to load.")
    # Import via importlib to avoid torch cpp_extension's build/load side effects.
    import importlib.util
    spec = importlib.util.spec_from_file_location("gsplat_cuda_p2_1a", proto_path)
    if spec is None or spec.loader is None:
        _fatal("could not build import spec for prototype .so")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # A smoke from the bound entry point namespace is acceptable: importing a
    # torch CUDA extension initialises a CUDA context but runs NO kernels.
    have = {n for n in dir(mod) if not n.startswith("_")}
    print(f"[P2-1A] prototype loaded; {len(have)} public symbols.")
    return mod


# ---------------------------------------------------------------------------
# Step 2: structural equivalence (host-side, no kernels)
# ---------------------------------------------------------------------------

def structural_equivalence(mod):
    _step(2, "structural equivalence")
    # Fingerprint the bound functions we require from the FP32 native hierarchy.
    required = {
        "higs_gatherless_projected_producer",
        "higs_camera_positions_from_viewmats",
        "higs_gather_rows",
        "higs_union_visible_mask",
    }
    have = {n for n in dir(mod) if not n.startswith("_")}
    missing = required - have
    if missing:
        _fatal(f"structural gate failed: missing bindings {missing}")
    print("[P2-1A] required FP32 producer bindings present.", flush=True)


# ---------------------------------------------------------------------------
# Steps 3-6 placeholders (correctness + timing body)
# ---------------------------------------------------------------------------

def ordering_equivalence(mod):
    _step(3, "ordering equivalence — NOT IMPLEMENTED ON NON-GPU PATH")
    # Placeholder. The future GPU run fills this in. Keep the abort gate honest:
    # if this body ever runs on this path it must be completed with the oracle.
    _fatal("ordering_equivalence body must be supplied by the P2-1A GPU author.")


def rgb_alpha_correctness(mod):
    _step(4, "RGB / alpha correctness — NOT IMPLEMENTED ON NON-GPU PATH")
    _fatal("rgb_alpha_correctness body must be supplied by the P2-1A GPU author.")


def resource_runtime_confirmation(mod):
    _step(5, "resource runtime confirmation — NOT IMPLEMENTED ON NON-GPU PATH")


def timing(mod):
    _step(6, "timing")
    # gated behind all prior correctness gates passing.
    print("[P2-1A] timing body to be supplied by the P2-1A GPU author.", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="P2-1A next-GPU execution package")
    p.add_argument("--proto", required=True, help="path to gsplat_cuda_p2_1a.so")
    p.add_argument("--skip-idle-check", action="store_true", help="DEBUG ONLY")
    args = p.parse_args(argv)

    if not args.skip_idle_check:
        check_gpus_idle()

    mod = load_prototype(args.proto)
    structural_equivalence(mod)
    # Correctness gates abort before timing:
    ordering_equivalence(mod)
    rgb_alpha_correctness(mod)
    resource_runtime_confirmation(mod)
    timing(mod)
    print("\n[P2-1A] all gates passed; timing completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
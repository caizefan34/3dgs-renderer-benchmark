#!/usr/bin/env python3
"""CUDA-event timing for the validation-only Macro-F4 prototype."""
import argparse
import importlib.util
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

from h3_fwd_0_oracle import f2_b2


def load_extensions(source, core_so):
    spec = importlib.util.spec_from_file_location("gsplat_cuda", core_so)
    core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
    sys.modules["gsplat.csrc"] = core
    build = runpy.run_path(str(Path(source) / "gsplat/experimental/render/kernels/cuda/build.py"))
    return build["build_and_load_experimental_gaussian_render_inference_scene"]()


def stats(samples):
    x = np.asarray(samples, dtype=np.float64)
    return {"median_ms": float(np.median(x)), "mean_ms": float(x.mean()),
            "p10_ms": float(np.percentile(x, 10)), "p90_ms": float(np.percentile(x, 90)),
            "std_ms": float(x.std()), "samples": int(len(x))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--source", required=True); ap.add_argument("--core-so", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--scene", default="room"); ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()
    ext = load_extensions(args.source, args.core_so)
    from gsplat.cuda._wrapper import isect_offset_encode
    f = f2_b2(args.scene, args.cam, 2048, "cuda")
    m2d = torch.from_numpy(f["m2d"]).cuda(); conics = torch.from_numpy(f["conics"]).cuda()
    depths = torch.from_numpy(f["depth"]).cuda(); opacity = torch.from_numpy(f["opacity"]).cuda(); radii = torch.from_numpy(f["radii"]).cuda()
    bm2d, bradii, bdepth, bconic = (x[None, None].contiguous() for x in (m2d, radii, depths, conics))
    bopacity = opacity[None, None].contiguous()
    def b2():
        try:
            _, isect, _ = torch.ops.gsplat.intersect_tile(bm2d, bradii, bdepth, bconic, bopacity, None, None, 1, 16, f["tw"], f["th"], True, False, None)
        except RuntimeError as e:
            if "expected at most 13" not in str(e): raise
            _, isect, _ = torch.ops.gsplat.intersect_tile(bm2d, bradii, bdepth, bconic, bopacity, None, None, 1, 16, f["tw"], f["th"], True, False)
        return isect_offset_encode(isect, 1, f["tw"], f["th"])
    def macro(profile=False):
        # Retained masks are production Macro-F4 state for the future F5 contract.
        return ext.higs_train_macro_f4(m2d, conics, depths, opacity, radii, f["tw"], f["th"], 16, True, profile)
    # Fixed protocol: 20 warmup, then five interleaved repetitions of 100 CUDA-event samples.
    for _ in range(20): b2(); macro(False)
    torch.cuda.synchronize()
    b2_samples, macro_samples, stage = [], [], [[], [], [], [], [], []]
    for _rep in range(5):
        for _ in range(100):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True); start.record(); b2(); end.record(); end.synchronize(); b2_samples.append(start.elapsed_time(end))
            start, end = torch.cuda.Event(True), torch.cuda.Event(True); start.record(); macro(False); end.record(); end.synchronize(); macro_samples.append(start.elapsed_time(end))
        # Stage events are internal and optional; profiling is isolated from total timing.
        for _ in range(100):
            values = macro(True)[4].numpy()
            for i, value in enumerate(values): stage[i].append(float(value))
    result = {"protocol": {"warmup": 20, "measurements_per_rep": 100, "repetitions": 5, "clock": "CUDA Events", "stream": "current CUDA stream", "interleaved": True},
              "B2_F4_total": stats(b2_samples), "Macro_F4_total": stats(macro_samples),
              "Macro_F4_stages": {name: stats(values) for name, values in zip(["count", "scan_prefix", "fill_exact_coverage", "segmented_sort", "batch_metadata", "retained_mask_generation"], stage)},
              "note": "Retained masks are enabled for both Macro-F4 totals and internal CUDA-event stage samples."}
    Path(args.out).write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()

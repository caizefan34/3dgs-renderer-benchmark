#!/usr/bin/env python3
"""Validation-only comparison of the CUDA FP32 Macro-F4 representation."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

from h3_fwd_0_oracle import MTW, MTH, f2_b2, oracle


def load_extensions(source, core_so):
    spec = importlib.util.spec_from_file_location("gsplat_cuda", core_so)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    sys.modules["gsplat.csrc"] = core
    build = runpy.run_path(str(Path(source) / "gsplat/experimental/render/kernels/cuda/build.py"))
    return build["build_and_load_experimental_gaussian_render_inference_scene"]()


def tile_lists(flat, offsets, tile_count):
    start = offsets.reshape(-1)
    end = np.append(start[1:], len(flat))
    return [list(map(int, flat[a:b])) for a, b in zip(start[:tile_count], end[:tile_count])]


def cuda_lists(outputs, tw, th):
    offsets, sorted_ids, batch_offsets, masks = (x.detach().cpu().numpy() for x in outputs)
    mw = math.ceil(tw / MTW)
    out = [[] for _ in range(tw * th)]
    for macro in range(len(offsets) - 1):
        mx, my = macro % mw, macro // mw
        for p in range(int(offsets[macro]), int(offsets[macro + 1])):
            gaussian = int(sorted_ids[p])
            mask = int(np.uint32(masks[p]))
            for bit in range(32):
                if not (mask & (1 << bit)):
                    continue
                tx, ty = mx * MTW + bit % MTW, my * MTH + bit // MTW
                if tx < tw and ty < th:
                    out[ty * tw + tx].append(gaussian)
    return out, offsets, sorted_ids, batch_offsets, masks


def compare(reference, reconstructed, depths):
    missing = extra = duplicate = wrong = inversions = ties = exact = 0
    order_rows = []
    pair_rows = []
    for tile, (ref, got) in enumerate(zip(reference, reconstructed)):
        rs, gs = set(ref), set(got)
        miss, ext = rs - gs, gs - rs
        dup = len(got) - len(gs)
        missing += len(miss); extra += len(ext); duplicate += dup; wrong += len(miss) + len(ext)
        if ref == got:
            exact += 1
            continue
        pos = {g: i for i, g in enumerate(got)}
        common = [g for g in ref if g in pos]
        inv = 0
        local_ties = 0
        for i, left in enumerate(common):
            for right in common[i + 1:]:
                if pos[left] > pos[right]:
                    inv += 1
                if depths[left] == depths[right]:
                    local_ties += 1
        inversions += inv; ties += local_ties
        pair_rows.append((tile, len(ref), len(got), len(miss), len(ext), dup))
        order_rows.append((tile, inv, local_ties, len(common)))
    return dict(
        missing_pairs=missing, extra_pairs=extra, duplicate_pairs=duplicate,
        wrong_gaussian_ids=wrong, exact_order_match_fraction=exact / len(reference),
        pairwise_inversions=inversions, equal_depth_ties=ties,
        pair_rows=pair_rows, order_rows=order_rows,
    )


def mask_stats(masks):
    pops = np.asarray([int(np.uint32(x)).bit_count() for x in masks], dtype=np.int32)
    return {"mean": float(pops.mean()), "p50": float(np.percentile(pops, 50)),
            "p90": float(np.percentile(pops, 90)), "p95": float(np.percentile(pops, 95)),
            "p99": float(np.percentile(pops, 99)), "max": int(pops.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True); ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--max-long-side", type=int, default=2048); ap.add_argument("--source", required=True)
    ap.add_argument("--core-so", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    selection = os.environ.get("HIGS_TRAIN_F4", "baseline")
    if selection not in {"baseline", "macro_fp32"}:
        raise RuntimeError("HIGS_TRAIN_F4 must be baseline or macro_fp32")
    # This is validation-only: F5 is intentionally not supplied with macro data.
    extension = load_extensions(args.source, args.core_so)
    f = f2_b2(args.scene, args.cam, args.max_long_side, "cuda")
    cpu = oracle(f)["stats"]
    inputs = [torch.from_numpy(f[k]).to("cuda") for k in ("m2d", "conics", "depth", "opacity", "radii")]
    outputs = extension.higs_train_macro_f4(*inputs, f["tw"], f["th"], 16, True, False)
    torch.cuda.synchronize()
    got, offsets, sorted_ids, batch_offsets, masks = cuda_lists(outputs[:4], f["tw"], f["th"])
    reference = tile_lists(f["flat"], f["offs"], f["tw"] * f["th"])
    cmp = compare(reference, got, f["depth"])
    n_fine = sum(map(len, got))
    n_macro = int(offsets[-1])
    result = {
        "scene": args.scene, "cam": args.cam, "width": f["width"], "height": f["height"],
        "selector": selection, "f5_source": "baseline", "N_B2_tile_pairs": len(f["flat"]),
        "N_macro_entries": n_macro, "N_reconstructed_fine_pairs": n_fine,
        "cpu_N_macro_entries": int(cpu["N_HiGS_macro_entries"]),
        "cpu_N_reconstructed_fine_pairs": int(cpu["N_HiGS_fine_pairs_after_masks"]),
        "cpu_cuda_macro_entries_equal": n_macro == int(cpu["N_HiGS_macro_entries"]),
        "cpu_cuda_fine_pairs_equal": n_fine == int(cpu["N_HiGS_fine_pairs_after_masks"]),
        "mask_popcount": mask_stats(masks), "representation_compression": len(f["flat"]) / n_macro,
        "macro_offsets_sha256": hashlib.sha256(offsets.tobytes()).hexdigest(),
        "macro_sorted_ids_sha256": hashlib.sha256(sorted_ids.tobytes()).hexdigest(),
        "macro_batch_offsets_sha256": hashlib.sha256(batch_offsets.tobytes()).hexdigest(),
        "validation_masks_sha256": hashlib.sha256(masks.tobytes()).hexdigest(),
        **{k: v for k, v in cmp.items() if not k.endswith("_rows")},
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / (args.scene + ".json")).write_text(json.dumps(result, indent=2))
    np.savetxt(out / (args.scene + "_pair_diff.csv"), np.asarray(cmp["pair_rows"], dtype=np.int64), fmt="%d", delimiter=",", header="tile,b2_count,macro_count,missing,extra,duplicates", comments="")
    np.savetxt(out / (args.scene + "_order.csv"), np.asarray(cmp["order_rows"], dtype=np.int64), fmt="%d", delimiter=",", header="tile,inversions,equal_depth_ties,common_count", comments="")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

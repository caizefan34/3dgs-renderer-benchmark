#!/usr/bin/env python3
"""P11 closure validation, streaming protocol (companion to Gate L3).

The L3 split-bench (sync per sample) systematically inflates graph-rebuilding
measurements: per-sample device syncs let GPU clocks sag and expose host-side
graph-rebuild gaps inside the event window. The trainer never syncs per
iteration, so the trainer-faithful protocol is STREAMING: enqueue N iterations
back-to-back with per-iteration event pairs, synchronize once at the end.

This script measures, in streaming mode at late_30k states:
  - renderer backward with external adjoint (rbwd; graph rebuilt per iter,
    rebuild enqueued untimed between event pairs)
  - loss-to-image backward (lbwd; retained graph, pure repeat)
  - full loss.backward() (full; graph rebuilt per iter, same as trainer)
and reports the additive closure rbwd + lbwd vs full.

Per-iteration event pairs with a rebuilt graph still bracket only the backward
kernels: the rebuild happens between ev1[i] and ev0[i+1] in stream order, so
elapsed(ev0[i], ev1[i]) contains only that iteration's backward kernels.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/epic05_p11")
import p11_common as C


def streaming_bench(pairs_fn, iters):
    """pairs_fn(i) -> (ev0, ev1) after enqueueing iteration i's bracketed work."""
    evs = []
    for i in range(iters):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        pairs_fn(i, ev0, ev1)
        evs.append((ev0, ev1))
    torch.cuda.synchronize()
    times = [a.elapsed_time(b) for a, b in evs]
    times.sort()
    n = len(times)
    med = times[n // 2] if n % 2 else 0.5 * (times[n // 2 - 1] + times[n // 2])
    return {"median_ms": med, "mean_ms": sum(times) / n,
            "min_ms": times[0], "p90_ms": times[int(0.9 * (n - 1))],
            "max_ms": times[-1], "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out",
                    default="/root/epic05_p11/dumps/P11_L3_CLOSURE_STREAMING.json")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    torch.manual_seed(0)

    C.bootstrap_gmain()
    C.import_recipe()
    gpu_rec = C.assert_clean_gpu(args.gpu)
    ssim_fn = C.SepSSIM(device="cuda")

    per_scene = {}
    for scene in C.SCENES:
        ckpt = os.path.join(C.FULL30K, f"gmain03_s42_{scene}", "ckpt_30000.pt")
        dataset = C.load_dataset(scene)
        model, _, meta = C.load_model_for_dataset(scene, dataset, ckpt_path=ckpt)
        seq = C.camera_sequence_for(scene)
        cam, gt = dataset.get_item(int(seq[0]))
        gt = gt.contiguous()

        # ---- external adjoint (exact, from a fresh graph) ----
        r0, image, radii_full, means2d, inputs = C.render_with_meta(
            model, cam, model.active_sh_degree, return_inputs=True)
        loss = 0.8 * F.l1_loss(image, gt) + 0.2 * ssim_fn(image, gt)
        v_image = torch.autograd.grad(loss, r0)[0].detach().contiguous()
        del r0, image, loss, means2d, inputs
        torch.cuda.synchronize()

        # ---- warmup all three paths ----
        for _ in range(args.warmup):
            r0, image, _, _, inputs = C.render_with_meta(
                model, cam, model.active_sh_degree, return_inputs=True)
            torch.autograd.grad(r0, inputs, grad_outputs=v_image)
        with torch.no_grad():
            r0u, _, _, _ = C.render_with_meta(model, cam, model.active_sh_degree)
        r0_leaf = r0u.detach().requires_grad_(True)
        loss_g = 0.8 * F.l1_loss(r0_leaf.clamp(0, 1), gt) + \
            0.2 * ssim_fn(r0_leaf.clamp(0, 1), gt)
        for _ in range(args.warmup):
            torch.autograd.grad(loss_g, r0_leaf, retain_graph=True)
        for _ in range(args.warmup):
            model.optimizer.zero_grad(set_to_none=True)
            r0, image, _, means2d = C.render_with_meta(
                model, cam, model.active_sh_degree)
            loss = 0.8 * F.l1_loss(image, gt) + 0.2 * ssim_fn(image, gt)
            loss.backward()
        torch.cuda.synchronize()

        # ---- 1. rbwd streaming (graph rebuilt per iter, untimed) ----
        def rbwd_pairs(i, ev0, ev1):
            r0, image, _, _, inputs = C.render_with_meta(
                model, cam, model.active_sh_degree, return_inputs=True)
            ev0.record()
            torch.autograd.grad(r0, inputs, grad_outputs=v_image)
            ev1.record()
        res_rbwd = streaming_bench(rbwd_pairs, args.iters)

        # ---- 2. lbwd streaming (retained graph, pure repeat) ----
        def lbwd_pairs(i, ev0, ev1):
            ev0.record()
            torch.autograd.grad(loss_g, r0_leaf, retain_graph=True)
            ev1.record()
        res_lbwd = streaming_bench(lbwd_pairs, args.iters)

        # ---- 3. full streaming (rebuild per iter, trainer-faithful) ----
        def full_pairs(i, ev0, ev1):
            model.optimizer.zero_grad(set_to_none=True)
            r0, image, _, means2d = C.render_with_meta(
                model, cam, model.active_sh_degree)
            loss = 0.8 * F.l1_loss(image, gt) + 0.2 * ssim_fn(image, gt)
            ev0.record()
            loss.backward()
            ev1.record()
        res_full = streaming_bench(full_pairs, args.iters)

        sum_ms = res_rbwd["median_ms"] + res_lbwd["median_ms"]
        gap_pct = abs(sum_ms - res_full["median_ms"]) / res_full["median_ms"] * 100
        per_scene[scene] = {
            "state": {"source": ckpt, "N": meta["state_N"]},
            "camera_index": int(seq[0]),
            "renderer_backward_external": res_rbwd,
            "loss_to_image_backward": res_lbwd,
            "full_backward": res_full,
            "rbwd_plus_lbwd_ms": sum_ms,
            "closure_gap_pct": gap_pct,
        }
        print(f"[{scene}] rbwd={res_rbwd['median_ms']:.2f} lbwd={res_lbwd['median_ms']:.2f} "
              f"sum={sum_ms:.2f} full={res_full['median_ms']:.2f} gap={gap_pct:.2f}%")

    record = {
        "phase": "P11 L3 companion - additive closure, streaming protocol",
        "protocol": {
            "iters": args.iters, "warmup": args.warmup,
            "design": "per-iteration event pairs, single synchronize at window "
                      "end (trainer-faithful: no per-sample device sync; graph "
                      "rebuilt per iteration outside the event brackets)",
            "rationale": "the L3 split-bench syncs per sample, which exposes "
                         "host-side rebuild gaps and clock sag inside the event "
                         "window; streaming matches the trainer's execution and "
                         "the campaign's phase-timing methodology",
        },
        "gpu": gpu_rec,
        "per_scene": per_scene,
    }
    C.dump_json(record, args.out)


if __name__ == "__main__":
    main()

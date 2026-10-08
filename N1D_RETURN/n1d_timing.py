#!/usr/bin/env python3
"""N1D timing: representative action timings, 10 warm-ups + 30 timed
repeats, CUDA events for phases + host wall for total, restore untimed,
no profiler, GPU otherwise idle. Per (scene, checkpoint, action):
anchor 1 provides v1 (and its A2 partner r1 for A2). A1-alpha cells share
A1's compute; they are timed anyway so the artifact shows alpha-invariance
of cost. Writes N1D_ACTION_TIMING_{scene}.json.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/root/epic05_n1")
import n1_core as C

N1D_ROOT = "/root/epic05_n1d"
N1D_ARTIFACTS = os.path.join(N1D_ROOT, "artifacts")
ALPHAS = (1.00, 0.75, 0.50, 0.25)
WARMUP = 10
REPEATS = 30


def timed(model, cam1, gt1, cam2, gt2, fused_loss, action, set_lrs=None):
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(8)]
    t0 = time.perf_counter()
    ev[0].record()
    if action == "A0":
        img, _, _ = C.render_view(model, cam1, model.active_sh_degree)
        ev[1].record()
        loss = fused_loss(img, gt1)
        ev[2].record()
        phases = {"forward": (0, 1), "loss": (1, 2)}
    elif action.startswith("A1"):
        img, _, m = C.render_view(model, cam1, model.active_sh_degree)
        ev[1].record()
        loss = fused_loss(img, gt1)
        ev[2].record()
        loss.backward()
        ev[3].record()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        ev[4].record()
        phases = {"forward": (0, 1), "loss": (1, 2), "backward": (2, 3),
                  "optimizer": (3, 4)}
    else:  # A2
        img1, _, _ = C.render_view(model, cam1, model.active_sh_degree)
        ev[1].record()
        loss1 = fused_loss(img1, gt1)
        ev[2].record()
        (loss1 * 0.5).backward()
        ev[3].record()
        img2, _, _ = C.render_view(model, cam2, model.active_sh_degree)
        ev[4].record()
        loss2 = fused_loss(img2, gt2)
        ev[5].record()
        (loss2 * 0.5).backward()
        ev[6].record()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        ev[7].record()
        phases = {"forward_1": (0, 1), "loss_1": (1, 2), "backward_1": (2, 3),
                  "forward_2": (3, 4), "loss_2": (4, 5), "backward_2": (5, 6),
                  "optimizer": (6, 7), "event_span_total": (0, 7)}
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1000.0
    ph = {k: ev[a].elapsed_time(ev[b]) for k, (a, b) in phases.items()}
    return wall, ph


def main():
    scene = sys.argv[1]
    partners = json.load(open(os.path.join(N1D_ARTIFACTS, "N1D_PARTNER_LIST.json")))
    P = partners["scenes"][scene]
    anchors_by_ckpt = {}
    for a in P["anchors"]:
        anchors_by_ckpt.setdefault(a["checkpoint"], []).append(a)

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.system(f"nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu "
              f"--format=csv,noheader > {N1D_ROOT}/cont_timing_{scene}_start.csv")

    C.bootstrap_c0()
    fused_loss = C.make_fused_loss()
    from dataset import GTDataset
    dataset = GTDataset(scene=scene, repo_root=C.REPO, resolution="1080p",
                        device="cuda", background="black")

    out = {"scene": scene, "gpu": gpu, "warmup": WARMUP, "repeats": REPEATS,
           "no_profiler": True, "restore_untimed": True, "cells": {}}
    for S in C.ANCHOR_CKPTS:
        a1 = anchors_by_ckpt[S][0]
        v1 = a1["v1"]
        pk = a1["n1d_partners"][0]
        cam1, gt1 = dataset.get_item(v1)
        cam2, gt2 = dataset.get_item(pk)
        ck = C.load_checkpoint(scene, S)
        model = C.build_model(ckpt=ck)
        C.scene_reset()
        del ck
        torch.cuda.empty_cache()
        master = C.StateMaster(model)
        model.update_learning_rate(S + 1)
        master_lrs = [g["lr"] for g in model.optimizer.param_groups]

        def set_lrs(alpha):
            for g, lr in zip(model.optimizer.param_groups, master_lrs):
                g["lr"] = lr * alpha

        cells = [("A0", 1.00), ("A1-1.00", 1.00), ("A1-0.75", 0.75),
                 ("A1-0.50", 0.50), ("A1-0.25", 0.25), ("A2-r1", 1.00)]
        for action, alpha in cells:
            walls, phase_acc = [], {}
            for rep in range(WARMUP + REPEATS):
                master.restore()
                model.update_learning_rate(S + 1)
                set_lrs(alpha)
                wall, ph = timed(model, cam1, gt1, cam2, gt2, fused_loss,
                                 action)
                set_lrs(1.00)
                if rep >= WARMUP:
                    walls.append(wall)
                    for k, v in ph.items():
                        phase_acc.setdefault(k, []).append(v)
            cell = {
                "anchor_id": 1, "v1": v1, "a2_partner": pk, "alpha": alpha,
                "total_wall_ms": {"median": float(np.median(walls)),
                                  "mean": float(np.mean(walls)),
                                  "min": float(np.min(walls)),
                                  "max": float(np.max(walls)),
                                  "std": float(np.std(walls))},
                "phases_ms": {k: {"median": float(np.median(v)),
                                  "mean": float(np.mean(v))}
                              for k, v in phase_acc.items()},
            }
            out["cells"][f"{S}_{action}"] = cell
            print(f"[{scene}@{S}] {action}: total median "
                  f"{cell['total_wall_ms']['median']:.3f} ms", flush=True)
        del model, master
        torch.cuda.empty_cache()

    os.system(f"nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu "
              f"--format=csv,noheader > {N1D_ROOT}/cont_timing_{scene}_end.csv")
    C.json_dump(out, os.path.join(N1D_ARTIFACTS, f"N1D_ACTION_TIMING_{scene}.json"))
    print(f"[{scene}] N1D timing COMPLETE")


if __name__ == "__main__":
    main()

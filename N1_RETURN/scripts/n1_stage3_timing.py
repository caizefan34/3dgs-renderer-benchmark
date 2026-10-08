#!/usr/bin/env python3
"""N1 Stage 3: action timing (10 warm-ups + 30 timed repeats, frozen state).

Per (scene, checkpoint, action type in {A0, A1, A2}): anchor 1 of that
checkpoint provides v1 (and its frozen A2-random partner for A2). Every
repeat starts from the bit-exact restored checkpoint state (restore is NOT
timed). CUDA events decompose forward/loss/backward/optimizer; host wall
gives the total. No profiler. GPU otherwise idle (contamination CSVs
captured around the run).

T(a) used by the analysis = median total wall of the 30 repeats.
Writes N1_ACTION_TIMING_{scene}.json.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n1_core as C

WARMUP = 10
REPEATS = 30


def timed_action(model, cam1, gt1, cam2, gt2, fused_loss, action):
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(8)]
    t0 = time.perf_counter()
    ev[0].record()
    if action == "A0":
        img, _, _ = C.render_view(model, cam1, model.active_sh_degree)
        ev[1].record()
        loss = fused_loss(img, gt1)
        ev[2].record()
        phases = {"forward": (0, 1), "loss": (1, 2)}
        n_ev = 3
    elif action == "A1":
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
        n_ev = 5
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
        phases = {"forward": (0, 1), "loss_1": (1, 2), "backward_1": (2, 3),
                  "forward_2": (3, 4), "loss_2": (4, 5), "backward_2": (5, 6),
                  "optimizer": (6, 7)}
        n_ev = 8
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1000.0
    ph = {k: ev[a].elapsed_time(ev[b]) for k, (a, b) in phases.items()}
    if action == "A2":
        ph["forward"] = ph.pop("forward") + ph.pop("forward_2")
        ph["loss"] = ph.pop("loss_1") + ph.pop("loss_2")
        ph["backward"] = ph.pop("backward_1") + ph.pop("backward_2")
    return wall, ph


def main():
    scene = sys.argv[1]
    probes = __import__("json").load(open(os.path.join(C.ARTIFACTS, "N1_PROBE_LIST.json")))
    P = probes["scenes"][scene]
    anchors_by_ckpt = {}
    for a in P["anchors"]:
        anchors_by_ckpt.setdefault(a["checkpoint"], []).append(a)

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.system(f"nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu "
              f"--format=csv,noheader > {C.N1_ROOT}/cont_timing_{scene}_start.csv")

    C.bootstrap_c0()
    fused_loss = C.make_fused_loss()
    from dataset import GTDataset
    dataset = GTDataset(scene=scene, repo_root=C.REPO, resolution="1080p",
                        device="cuda", background="black")

    out = {"scene": scene, "gpu": gpu, "warmup": WARMUP, "repeats": REPEATS,
           "no_profiler": True, "cells": {}}
    for S in C.ANCHOR_CKPTS:
        a1 = anchors_by_ckpt[S][0]  # anchor 1 (frozen)
        v1 = a1["v1"]
        pr = a1["a2_random_partner"]
        cam1, gt1 = dataset.get_item(v1)
        cam2, gt2 = dataset.get_item(pr)
        ck = C.load_checkpoint(scene, S)
        model = C.build_model(ckpt=ck)
        C.scene_reset()
        del ck
        torch.cuda.empty_cache()
        master = C.StateMaster(model)

        for action in ("A0", "A1", "A2"):
            walls, phase_acc = [], {}
            for rep in range(WARMUP + REPEATS):
                master.restore()
                model.update_learning_rate(S + 1)
                wall, ph = timed_action(model, cam1, gt1, cam2, gt2,
                                        fused_loss, action)
                if rep >= WARMUP:
                    walls.append(wall)
                    for k, v in ph.items():
                        phase_acc.setdefault(k, []).append(v)
            cell = {
                "anchor_id": 1, "v1": v1, "a2_partner": pr,
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
                  f"{cell['total_wall_ms']['median']:.3f} ms "
                  f"(fwd {cell['phases_ms']['forward']['median']:.3f})", flush=True)
        del model, master
        torch.cuda.empty_cache()

    os.system(f"nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu "
              f"--format=csv,noheader > {C.N1_ROOT}/cont_timing_{scene}_end.csv")
    C.json_dump(out, os.path.join(C.ARTIFACTS, f"N1_ACTION_TIMING_{scene}.json"))
    print(f"[{scene}] stage3 timing COMPLETE")


if __name__ == "__main__":
    main()

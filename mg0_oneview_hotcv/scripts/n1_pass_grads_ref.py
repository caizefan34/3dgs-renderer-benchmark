#!/usr/bin/env python3
"""N1 Stage 2a/2b: per-view true gradients, pair conflict metrics, sketches.

Per scene, per checkpoint S in {10000, 15000, 20000, 25000}:

Phase A (sketch + per-view loss): iterate the UNIQUE probe-view union of S
in frozen order; per view: forward + fused loss + backward at the restored
checkpoint state (no update); record the view's current fused loss; capture
its true gradient grouped as (xyz; opacity; scale+rotation; SH); flush
batches of 32 views through the globally-frozen dense Rademacher sketcher
(256 rows/group); store sketches (unit-normalized per group, concatenated
1024-dim) to N1_SKETCHES_{scene}.npz and per-view losses to
N1_VIEW_LOSSES_{scene}.json.

Phase B (conflict ground truth, anchor checkpoints only): per anchor,
recompute the true gradients of v1 and its 7 pool partners at the checkpoint
state; record per-group and global-normalized cosines, per-group norm
ratios, and visibility Jaccard for each (v1, partner) pair.

No state is ever updated in this pass. Output rows go to
N1_GRADIENT_CONFLICT_{scene}.json (merged to parquet later).
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n1_core as C

BATCH = 16  # sketch flush size (memory safety on bicycle: peak ~38GB)


def view_grad_and_loss(model, cam, gt, fused_loss):
    model.optimizer.zero_grad(set_to_none=True)
    img, radii, means2d = C.render_view(model, cam, model.active_sh_degree)
    loss = fused_loss(img, gt)
    loss.backward()
    torch.cuda.synchronize()
    vis = (radii > 0).any(dim=-1).reshape(-1)
    grads = C.capture_flat_grads(model)
    return float(loss.item()), grads, vis


def main():
    scene = sys.argv[1]
    probes = __import__("json").load(open(os.path.join(C.ARTIFACTS, "N1_PROBE_LIST.json")))
    P = probes["scenes"][scene]
    anchors_by_ckpt = {}
    for a in P["anchors"]:
        anchors_by_ckpt.setdefault(a["checkpoint"], []).append(a)
    union_by_ckpt = {int(k): v for k, v in P["probe_view_union_by_checkpoint"].items()}

    C.bootstrap_c0()
    fused_loss = C.make_fused_loss()
    from dataset import GTDataset
    dataset = GTDataset(scene=scene, repo_root=C.REPO, resolution="1080p",
                        device="cuda", background="black")

    sketcher = C.Sketcher()
    sketches = {}      # (ckpt, view) -> np.float32[1024]
    view_losses = {}   # str(ckpt) -> {view: loss}
    conflict_rows = []
    t_start = time.time()

    for S in C.CKPTS:
        t0 = time.time()
        ck = C.load_checkpoint(scene, S)
        model = C.build_model(ckpt=ck)
        C.scene_reset()
        del ck
        torch.cuda.empty_cache()

        # ---------------- Phase A: unique-union sketches + losses ----------
        uni = list(union_by_ckpt[S])
        losses_S = {}
        buf_views, buf_grads, buf_vis = [], [], []
        n_flush = 0

        def flush():
            nonlocal buf_views, buf_grads, buf_vis, n_flush
            if not buf_grads:
                return
            per_group_sk = {}
            for gname, _ in C.GROUPS:
                G, gidx = C.grads_to_group_matrix(buf_grads, gname)
                per_group_sk[gname] = sketcher.sketch_batch(gidx, G)
                del G
            Ssk = C.normalize_sketches(per_group_sk).cpu().numpy()
            for i, v in enumerate(buf_views):
                sketches[(S, int(v))] = Ssk[i].astype(np.float32)
            n_flush += 1
            buf_views, buf_grads, buf_vis = [], [], []

        for j, v in enumerate(uni):
            cam, gt = dataset.get_item(int(v))
            loss, grads, vis = view_grad_and_loss(model, cam, gt, fused_loss)
            losses_S[str(int(v))] = loss
            buf_views.append(int(v))
            buf_grads.append(grads)
            buf_vis.append(vis)
            if len(buf_grads) >= BATCH:
                flush()
            if (j + 1) % 64 == 0:
                print(f"  [{scene}@{S}] phaseA {j+1}/{len(uni)} "
                      f"flushes={n_flush} mem={C.gpu_mem_gb():.1f}GB", flush=True)
        flush()
        # EMA chain coverage: forward-only losses at this state for the v1
        # views of anchors at LATER checkpoints (spec: EMA needs L(v1) at
        # every snapshot <= C; union(S) only covers anchors at S and S+5000).
        uni_set = set(int(v) for v in uni)
        extra = sorted({a["v1"] for Ck in C.ANCHOR_CKPTS if Ck > S
                        for a in anchors_by_ckpt.get(Ck, [])} - uni_set)
        with torch.no_grad():
            for v in extra:
                cam, gt = dataset.get_item(v)
                img, _, _ = C.render_view(model, cam, model.active_sh_degree)
                losses_S[str(v)] = float(fused_loss(img, gt).item())
        if extra:
            print(f"  [{scene}@{S}] future-anchor v1 losses: +{len(extra)}",
                  flush=True)
        view_losses[str(S)] = losses_S
        del buf_grads

        # ---------------- Phase B: per-anchor pair conflict -----------------
        if S in C.ANCHOR_CKPTS:
            for ai, a in enumerate(anchors_by_ckpt[S]):
                v1 = a["v1"]
                cam, gt = dataset.get_item(v1)
                l1, g1, vis1 = view_grad_and_loss(model, cam, gt, fused_loss)
                pool_ids = [p["partner"] for p in a["pool"]]
                gpool, vispool = [], []
                for pk in pool_ids:
                    camp, gtp = dataset.get_item(pk)
                    lp, gp, visp = view_grad_and_loss(model, camp, gtp, fused_loss)
                    gpool.append(gp)
                    vispool.append(visp)
                for p, gp, visp in zip(a["pool"], gpool, vispool):
                    m = C.pair_grad_metrics(g1, gp)
                    conflict_rows.append({
                        "scene": scene, "checkpoint": S,
                        "anchor_id": a["anchor_index"], "iteration": a["iteration"],
                        "v1": v1, "partner": p["partner"], "tier": p["tier"],
                        "overlap_score": p["overlap_score"],
                        "cos_dir": p["cos_dir"], "dist": p["dist"],
                        "global_cosine": m["global_cosine"],
                        "global_norm_a": m["global_norm_a"],
                        "global_norm_b": m["global_norm_b"],
                        "cos_xyz": m["per_group"]["xyz"]["cosine"],
                        "cos_opacity": m["per_group"]["opacity"]["cosine"],
                        "cos_scale_rot": m["per_group"]["scale_rot"]["cosine"],
                        "cos_sh": m["per_group"]["sh"]["cosine"],
                        "norm_ratio_xyz": m["per_group"]["xyz"]["norm_ratio_b_over_a"],
                        "norm_ratio_global": (m["global_norm_b"] / m["global_norm_a"]
                                              if m["global_norm_a"] > 0 else float("nan")),
                        "vis_jaccard": C.jaccard(vis1, visp),
                        "vis_intersection": int((vis1 & visp).sum()),
                        "vis_union": int((vis1 | visp).sum()),
                        "v1_loss": l1,
                    })
                C.free_flat_grads(g1)
                for gp in gpool:
                    C.free_flat_grads(gp)
                del gpool, vispool, g1
                if (ai + 1) % 16 == 0:
                    print(f"  [{scene}@{S}] phaseB anchors {ai+1}/64 "
                          f"mem={C.gpu_mem_gb():.1f}GB", flush=True)

        del model
        torch.cuda.empty_cache()
        print(f"[{scene}@{S}] done in {time.time()-t0:.0f}s "
              f"(union={len(uni)}, sketches={sum(1 for k in sketches if k[0]==S)}, "
              f"conflict_rows={sum(1 for r in conflict_rows if r['checkpoint']==S)})",
              flush=True)

    # ---- store ---------------------------------------------------------
    sk = {f"{k[0]}_{k[1]}": v for k, v in sketches.items()}
    np.savez_compressed(os.path.join(C.ARTIFACTS, f"N1_SKETCHES_{scene}.npz"), **sk)
    C.json_dump({"scene": scene, "per_view_fused_loss_by_ckpt": view_losses},
                os.path.join(C.ARTIFACTS, f"N1_VIEW_LOSSES_{scene}.json"))
    C.json_dump({"scene": scene, "rows": conflict_rows},
                os.path.join(C.ARTIFACTS, f"N1_GRADIENT_CONFLICT_{scene}.json"))
    print(f"[{scene}] pass_grads COMPLETE in {(time.time()-t_start)/60:.1f} min; "
          f"rows={len(conflict_rows)} sketches={len(sketches)}")


if __name__ == "__main__":
    main()

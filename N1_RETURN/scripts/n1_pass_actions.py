#!/usr/bin/env python3
"""N1 Stage 2c: counterfactual action measurements.

Per scene, per anchor checkpoint C in {15000, 20000, 25000}, per anchor:
  A0        : forward + fused loss on v1 (gating basis), NO update;
              held-out re-evaluated literally.
  A1 x3     : single-view backward + Adam step (replicas: execution noise).
  A2_random : equal-weight two-view minibatch with the frozen deterministic
              partner: render v1, (0.5*L).backward(); render partner,
              (0.5*L).backward(); ONE optimizer step.
  A2_best   : the A2 action with EACH of the 7 pool partners (diagnostic
              oracle sweep; best-by-E selected later in analysis).

Every action starts from the bit-exact restored checkpoint state
(StateMaster; update_learning_rate(C+1) reproduces training iteration C+1).
Held-out progress is measured on the frozen 8-camera set immediately after
each action. The live canonical checkpoints are never touched.

Writes N1_ACTION_RESULTS_{scene}.json (merged to parquet in analysis).
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n1_core as C


def action_A0(model, cam, gt, fused_loss):
    """Forward + loss only (the information a gating policy would compute)."""
    t0 = time.perf_counter()
    img, _, _ = C.render_view(model, cam, model.active_sh_degree)
    loss = fused_loss(img, gt)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1000.0
    return {"loss_v1": float(loss.item()), "wall_ms": wall}


def action_A1(model, cam, gt, fused_loss):
    t0 = time.perf_counter()
    img, _, means2d = C.render_view(model, cam, model.active_sh_degree)
    loss = fused_loss(img, gt)
    loss.backward()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1000.0
    return {"loss_v1": float(loss.item()), "wall_ms": wall}


def action_A2(model, cam1, gt1, cam2, gt2, fused_loss):
    """Equal-weight two-view minibatch: one joint update."""
    t0 = time.perf_counter()
    img1, _, m1 = C.render_view(model, cam1, model.active_sh_degree)
    loss1 = fused_loss(img1, gt1)
    (loss1 * 0.5).backward()
    img2, _, m2 = C.render_view(model, cam2, model.active_sh_degree)
    loss2 = fused_loss(img2, gt2)
    (loss2 * 0.5).backward()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1000.0
    return {"loss_v1": float(loss1.item()), "loss_v2": float(loss2.item()),
            "wall_ms": wall}


def main():
    scene = sys.argv[1]
    probes = __import__("json").load(open(os.path.join(C.ARTIFACTS, "N1_PROBE_LIST.json")))
    P = probes["scenes"][scene]
    anchors_by_ckpt = {}
    for a in P["anchors"]:
        anchors_by_ckpt.setdefault(a["checkpoint"], []).append(a)
    heldout_ids = P["heldout_cameras"]

    C.bootstrap_c0()
    fused_loss = C.make_fused_loss()
    from dataset import GTDataset
    dataset = GTDataset(scene=scene, repo_root=C.REPO, resolution="1080p",
                        device="cuda", background="black")
    heldout_cams_gts = [dataset.get_item(i) for i in heldout_ids]

    rows = []
    t_start = time.time()
    for S in C.ANCHOR_CKPTS:
        t0 = time.time()
        ck = C.load_checkpoint(scene, S)
        model = C.build_model(ckpt=ck)
        C.scene_reset()
        del ck
        torch.cuda.empty_cache()
        master = C.StateMaster(model)

        hb = C.heldout_eval(model, heldout_cams_gts, fused_loss)
        print(f"[{scene}@{S}] heldout_before: {hb}", flush=True)

        for a in anchors_by_ckpt[S]:
            v1 = a["v1"]
            cam1, gt1 = dataset.get_item(v1)
            base = {"scene": scene, "checkpoint": S, "anchor_id": a["anchor_index"],
                    "iteration": a["iteration"], "v1": v1,
                    "heldout_fused_before": hb["fused"],
                    "heldout_l1_before": hb["l1"], "heldout_psnr_before": hb["psnr"]}

            # ---- A0 -----------------------------------------------------
            master.restore()
            model.update_learning_rate(S + 1)
            r = action_A0(model, cam1, gt1, fused_loss)
            ha = C.heldout_eval(model, heldout_cams_gts, fused_loss)
            rows.append({**base, "action": "A0", "partner": None,
                         "heldout_fused_after": ha["fused"],
                         "heldout_l1_after": ha["l1"],
                         "heldout_psnr_after": ha["psnr"],
                         "delta_fused": hb["fused"] - ha["fused"],
                         "delta_l1": hb["l1"] - ha["l1"],
                         "loss_v1": r["loss_v1"], "loss_v2": None,
                         "wall_ms": r["wall_ms"],
                         "upd_norm_global": 0.0, "upd_norm_xyz": 0.0})

            # ---- A1 replicas ---------------------------------------------
            for k in range(C.A1_REPLICAS):
                master.restore()
                model.update_learning_rate(S + 1)
                r = action_A1(model, cam1, gt1, fused_loss)
                un = master.param_update_norm()
                ha = C.heldout_eval(model, heldout_cams_gts, fused_loss)
                rows.append({**base, "action": f"A1_r{k+1}", "partner": None,
                             "heldout_fused_after": ha["fused"],
                             "heldout_l1_after": ha["l1"],
                             "heldout_psnr_after": ha["psnr"],
                             "delta_fused": hb["fused"] - ha["fused"],
                             "delta_l1": hb["l1"] - ha["l1"],
                             "loss_v1": r["loss_v1"], "loss_v2": None,
                             "wall_ms": r["wall_ms"],
                             "upd_norm_global": un["global"], "upd_norm_xyz": un["xyz"]})

            # ---- A2 random -------------------------------------------------
            pr = a["a2_random_partner"]
            cam2, gt2 = dataset.get_item(pr)
            master.restore()
            model.update_learning_rate(S + 1)
            r = action_A2(model, cam1, gt1, cam2, gt2, fused_loss)
            un = master.param_update_norm()
            ha = C.heldout_eval(model, heldout_cams_gts, fused_loss)
            rows.append({**base, "action": "A2_random", "partner": pr,
                         "heldout_fused_after": ha["fused"],
                         "heldout_l1_after": ha["l1"],
                         "heldout_psnr_after": ha["psnr"],
                         "delta_fused": hb["fused"] - ha["fused"],
                         "delta_l1": hb["l1"] - ha["l1"],
                         "loss_v1": r["loss_v1"], "loss_v2": r["loss_v2"],
                         "wall_ms": r["wall_ms"],
                         "upd_norm_global": un["global"], "upd_norm_xyz": un["xyz"]})

            # ---- A2 oracle sweep (7 partners) -------------------------------
            for p in a["pool"]:
                pk = p["partner"]
                cam2, gt2 = dataset.get_item(pk)
                master.restore()
                model.update_learning_rate(S + 1)
                r = action_A2(model, cam1, gt1, cam2, gt2, fused_loss)
                un = master.param_update_norm()
                ha = C.heldout_eval(model, heldout_cams_gts, fused_loss)
                rows.append({**base, "action": "A2_pool",
                             "partner": pk,
                             "heldout_fused_after": ha["fused"],
                             "heldout_l1_after": ha["l1"],
                             "heldout_psnr_after": ha["psnr"],
                             "delta_fused": hb["fused"] - ha["fused"],
                             "delta_l1": hb["l1"] - ha["l1"],
                             "loss_v1": r["loss_v1"], "loss_v2": r["loss_v2"],
                             "wall_ms": r["wall_ms"],
                             "upd_norm_global": un["global"], "upd_norm_xyz": un["xyz"]})

            if a["anchor_index"] % 16 == 0:
                el = time.time() - t0
                print(f"  [{scene}@{S}] anchor {a['anchor_index']}/64 "
                      f"rows={len(rows)} elapsed={el:.0f}s mem={C.gpu_mem_gb():.1f}GB",
                      flush=True)

        del model, master
        torch.cuda.empty_cache()
        print(f"[{scene}@{S}] done in {time.time()-t0:.0f}s", flush=True)

    C.json_dump({"scene": scene, "rows": rows},
                os.path.join(C.ARTIFACTS, f"N1_ACTION_RESULTS_{scene}.json"))
    print(f"[{scene}] pass_actions COMPLETE in {(time.time()-t_start)/60:.1f} min; "
          f"rows={len(rows)}")


if __name__ == "__main__":
    main()

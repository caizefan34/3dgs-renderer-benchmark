#!/usr/bin/env python3
"""N1D Stage B: counterfactual actions (raw rows only; no analysis).

Per scene, per anchor checkpoint C in {15000, 20000, 25000}, per anchor:
  A0        : forward + fused loss on v1 (no update); held-out re-evaluated.
  A1-1.00   : original single-view C0 action.
  A1-0.75 / A1-0.50 / A1-0.25 : identical A1 gradient and Adam semantics,
              one-step parameter-group lr multiplier alpha, then the frozen
              schedule restored. Adam moments/beta/eps/step untouched.
  A2-r1/r2/r3 : two true full-resolution view gradients (v1 + frozen partner
              k), equal-weight mean, ONE ordinary Adam update (lr x1.00).

Every action starts from the bit-exact restored N1 checkpoint state
(StateMaster). Held-out = the exact N1 8-camera set; DeltaL = fused_before -
fused_after. Single-shot host wall (ms) recorded per action (restore and
held-out evaluation excluded). No A2-best search. No replicas (one row per
action, as the frozen analyzer expects).

Writes N1D_ACTION_RESULTS_{scene}.json (merged to parquet in finalize).
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/root/epic05_n1")
import n1_core as C

N1D_ROOT = "/root/epic05_n1d"
N1D_ARTIFACTS = os.path.join(N1D_ROOT, "artifacts")
N1D_SEED = 20260929
ALPHAS = (1.00, 0.75, 0.50, 0.25)


class N1DMaster(C.StateMaster):
    """StateMaster + absolute one-step LR scaling for the damping controls.

    master_lrs must be (re)captured after the caller runs
    update_learning_rate(C+1): capture() does that.
    """

    def capture(self):
        self.master_lrs = [g["lr"] for g in self.model.optimizer.param_groups]

    def set_lrs(self, alpha):
        for g, lr in zip(self.model.optimizer.param_groups, self.master_lrs):
            g["lr"] = lr * alpha

    def check_lrs_baseline(self):
        for g, lr in zip(self.model.optimizer.param_groups, self.master_lrs):
            assert g["lr"] == lr, "LR invariant violated"


def action_A0(model, cam, gt, fused_loss):
    t0 = time.perf_counter()
    img, _, _ = C.render_view(model, cam, model.active_sh_degree)
    loss = fused_loss(img, gt)
    torch.cuda.synchronize()
    return {"loss_v1": float(loss.item()),
            "wall_ms": (time.perf_counter() - t0) * 1000.0}


def action_A1(model, cam, gt, fused_loss):
    t0 = time.perf_counter()
    img, _, means2d = C.render_view(model, cam, model.active_sh_degree)
    loss = fused_loss(img, gt)
    loss.backward()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return {"loss_v1": float(loss.item()),
            "wall_ms": (time.perf_counter() - t0) * 1000.0}


def action_A2(model, cam1, gt1, cam2, gt2, fused_loss):
    t0 = time.perf_counter()
    img1, _, _ = C.render_view(model, cam1, model.active_sh_degree)
    loss1 = fused_loss(img1, gt1)
    (loss1 * 0.5).backward()
    img2, _, _ = C.render_view(model, cam2, model.active_sh_degree)
    loss2 = fused_loss(img2, gt2)
    (loss2 * 0.5).backward()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return {"loss_v1": float(loss1.item()), "loss_v2": float(loss2.item()),
            "wall_ms": (time.perf_counter() - t0) * 1000.0}


def main():
    scene = sys.argv[1]
    partners = json.load(open(os.path.join(N1D_ARTIFACTS, "N1D_PARTNER_LIST.json")))
    P = partners["scenes"][scene]
    heldout_ids = P["heldout_cameras"]
    anchors_by_ckpt = {}
    for a in P["anchors"]:
        anchors_by_ckpt.setdefault(a["checkpoint"], []).append(a)

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
        master = N1DMaster(model)
        model.update_learning_rate(S + 1)
        master.capture()
        lr0 = master.master_lrs[0]

        hb = C.heldout_eval(model, heldout_cams_gts, fused_loss)
        print(f"[{scene}@{S}] heldout_before: {hb} (xyz lr={lr0:.3e})", flush=True)

        for a in anchors_by_ckpt[S]:
            v1 = a["v1"]
            cam1, gt1 = dataset.get_item(v1)
            base = {"scene": scene, "checkpoint": S, "anchor_id": a["anchor_id"],
                    "iteration": a["iteration"], "v1": v1,
                    "heldout_fused_before": hb["fused"],
                    "heldout_l1_before": hb["l1"],
                    "heldout_psnr_before": hb["psnr"]}

            def run(tag, alpha, fn, partner=None, cam2=None, gt2=None):
                lr_alpha = alpha if alpha is not None else 1.00
                master.restore()
                model.update_learning_rate(S + 1)  # idempotent schedule
                master.set_lrs(lr_alpha)
                r = fn(model, cam1, gt1, cam2, gt2, fused_loss) \
                    if cam2 is not None else fn(model, cam1, gt1, fused_loss)
                master.set_lrs(1.00)
                master.check_lrs_baseline()
                ha = C.heldout_eval(model, heldout_cams_gts, fused_loss)
                un = master.param_update_norm()
                rows.append({**base, "action": tag, "alpha": alpha,
                             "partner": partner,
                             "heldout_fused_after": ha["fused"],
                             "heldout_l1_after": ha["l1"],
                             "heldout_psnr_after": ha["psnr"],
                             "delta_fused": hb["fused"] - ha["fused"],
                             "delta_l1": hb["l1"] - ha["l1"],
                             "loss_v1": r["loss_v1"],
                             "loss_v2": r.get("loss_v2"),
                             "wall_ms": r["wall_ms"],
                             "upd_norm_global": un["global"],
                             "upd_norm_xyz": un["xyz"]})

            run("A0", None, action_A0)
            for alpha in ALPHAS:
                run(f"A1-{alpha:.2f}", alpha, action_A1)
            for k in (1, 2, 3):
                pk = a["n1d_partners"][k - 1]
                cam2, gt2 = dataset.get_item(pk)
                run(f"A2-r{k}", None, action_A2, partner=pk,
                    cam2=cam2, gt2=gt2)
                del cam2, gt2

            if a["anchor_id"] % 16 == 0:
                print(f"  [{scene}@{S}] anchor {a['anchor_id']}/64 "
                      f"rows={len(rows)} elapsed={time.time()-t0:.0f}s "
                      f"mem={C.gpu_mem_gb():.1f}GB", flush=True)

        del model, master
        torch.cuda.empty_cache()
        print(f"[{scene}@{S}] done in {time.time()-t0:.0f}s", flush=True)

    C.json_dump({"scene": scene, "n1d_seed": N1D_SEED, "rows": rows},
                os.path.join(N1D_ARTIFACTS, f"N1D_ACTION_RESULTS_{scene}.json"))
    print(f"[{scene}] N1D actions COMPLETE in {(time.time()-t_start)/60:.1f} "
          f"min; rows={len(rows)}")


if __name__ == "__main__":
    main()

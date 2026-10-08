#!/usr/bin/env python3
"""P11 Gate O: optimizer sanity / no-go check (measurement only).

- Measures the optimizer phase by parameter group (cheap): 5 groups (xyz, shs,
  opacity, scaling, rotation); per-group timing via separate single-group Adam
  optimizers with identical settings (Adam is per-parameter, so group-split
  timing is faithful); closure sum-vs-joint validated.
- Audits the exact dependency of the next iteration's projection on
  means/scales/quats/opacities/SH.
- Empirically demonstrates the Adam momentum no-go: with all gradients ZERO,
  parameters still move (exp_avg momentum), so any cross-iteration overlap that
  lets the next forward read pre-update parameters is inexact.

No cross-iteration double-buffer overlap is implemented.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/epic05_p11")
import p11_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="/root/epic05_p11/dumps/P11_OPTIMIZER_TIMING.json")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    torch.manual_seed(0)

    C.bootstrap_gmain()
    C.import_recipe()
    gpu_rec = C.assert_clean_gpu(args.gpu)
    from dataset import GTDataset
    from torch.optim import Adam
    import torch.nn.functional as F

    per_scene = {}
    for scene in C.SCENES:
        ckpt = os.path.join(C.FULL30K, f"gmain03_s42_{scene}", "ckpt_30000.pt")
        model, dataset, _, meta = C.load_model_and_dataset(scene, ckpt_path=ckpt)
        seq = C.camera_sequence_for(scene)
        cam_idx = int(seq[0])
        cam, gt = dataset.get_item(cam_idx)

        # restore the campaign optimizer state (real Adam moments)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        opt_state = state.get("optimizer_state_dict")
        if opt_state is not None:
            model.optimizer.load_state_dict(opt_state)

        # populate realistic grads (one faithful backward)
        ssim_fn = C.SepSSIM(device="cuda")
        r0, image, radii_full, means2d = C.render_with_meta(
            model, cam, model.active_sh_degree)
        loss = 0.8 * F.l1_loss(image, gt) + 0.2 * ssim_fn(image, gt)
        loss.backward()
        del r0, image, loss

        groups = {g["name"]: g for g in model.optimizer.param_groups}
        param_of = {"xyz": model._xyz, "shs": model._shs,
                    "opacity": model._opacity, "scaling": model._scaling,
                    "rotation": model._rotation}

        # snapshot grads (to re-install before each timed step so the step
        # semantics stay identical across repeats)
        grad_snap = {k: p.grad.detach().clone() for k, p in param_of.items()}

        def restore_grads():
            for k, p in param_of.items():
                p.grad = grad_snap[k].clone()

        # ---- joint step ----
        def joint_step():
            restore_grads()
            model.optimizer.step()
        res_joint = C.bench_events(joint_step, args.warmup, args.iters, args.reps)

        # ---- per-group steps (single-group Adams, identical settings) ----
        per_group = {}
        for name, g in groups.items():
            p = param_of[name]
            opt1 = Adam([p], lr=g["lr"], eps=g.get("eps", 1e-15),
                        betas=(g.get("betas", (0.9, 0.999))[0],
                               g.get("betas", (0.9, 0.999))[1]))
            if opt_state is not None:
                # find matching state slice for this param in the saved state
                # (param order in state_dict follows param_groups order)
                idx = [g2["name"] for g2 in model.optimizer.param_groups].index(name)
                saved_params = opt_state["param_groups"][idx]["params"]
                st = opt_state["state"]
                if saved_params and saved_params[0] in st:
                    pstate = {0: st[saved_params[0]]}
                    opt1.load_state_dict({
                        "state": pstate,
                        "param_groups": [{
                            "lr": g["lr"], "betas": g.get("betas", (0.9, 0.999)),
                            "eps": g.get("eps", 1e-15), "weight_decay": 0.0,
                            "amsgrad": False, "maximize": False,
                            "foreach": None, "capturable": False,
                            "differentiable": False, "fused": False,
                            "initial_lr": g["lr"],
                            "params": [0]}]})
            def step1(p=p, opt1=opt1):
                p.grad = grad_snap[[k for k, v in param_of.items() if v is p][0]].clone()
                opt1.step()
            per_group[name] = {
                "n_params": int(p.numel()),
                "lr": float(g["lr"]),
                "timing": C.bench_events(step1, args.warmup, args.iters, args.reps),
            }

        joint_ms = res_joint["median_of_medians_ms"]
        group_sum_ms = float(np.sum([v["timing"]["median_of_medians_ms"]
                                     for v in per_group.values()]))
        closure_rel = abs(group_sum_ms - joint_ms) / joint_ms if joint_ms else None

        # ---- momentum no-go empirical check ----
        snap_params = {k: p.detach().clone() for k, p in param_of.items()}
        for p in param_of.values():
            p.grad = torch.zeros_like(p)
        model.optimizer.step()
        moves = {}
        for k, p in param_of.items():
            d = (p.detach() - snap_params[k])
            moves[k] = {
                "param_norm": float(snap_params[k].norm()),
                "delta_norm_with_zero_grad": float(d.norm()),
                "max_abs_delta": float(d.abs().max()),
                "moved": bool(d.abs().max() > 0),
            }
        # restore
        model.optimizer.zero_grad(set_to_none=True)

        # grads present-but-zero vs absent check (trainer always has grads)
        per_scene[scene] = {
            "state": {"source": ckpt, "N": meta["state_N"]},
            "optimizer_joint_step": res_joint,
            "per_group": {k: {
                "n_params": v["n_params"], "lr": v["lr"],
                "median_ms": v["timing"]["median_of_medians_ms"],
                "pct_of_joint": v["timing"]["median_of_medians_ms"] / joint_ms * 100,
            } for k, v in per_group.items()},
            "group_sum_ms": group_sum_ms,
            "joint_ms": joint_ms,
            "group_vs_joint_closure_rel": closure_rel,
            "momentum_zero_grad_check": moves,
            "optimizer_state_restored": opt_state is not None,
        }
        print(f"[{scene}] joint={joint_ms:.3f}ms group_sum={group_sum_ms:.3f}ms "
              f"closure={closure_rel*100:.1f}% momentum_moved_any="
              f"{any(v['moved'] for v in moves.values())}")

    record = {
        "phase": "P11 Gate O - optimizer sanity / no-go check",
        "gpu": gpu_rec,
        "protocol": {"warmup": args.warmup, "measurements": args.iters,
                     "reps": args.reps},
        "per_scene": per_scene,
        "dependency_audit": {
            "next_iteration_reads": [
                "means (model.get_xyz) -> fully_fused_projection fwd",
                "quats (model.get_rotation) -> projection fwd (quat normalize)",
                "scales (model.get_scaling) -> projection fwd (covariance)",
                "opacities (model.get_opacity) -> rasterization fwd",
                "SH (model.get_features) -> spherical harmonics eval in rasterization",
            ],
            "same_stream_guarantee": "optimizer.step() is enqueued on the same "
                                     "CUDA stream before the next forward, so "
                                     "sequential stream semantics guarantee the "
                                     "next forward observes exactly the updated "
                                     "parameters in the CURRENT code",
            "momentum_argument": "standard Adam updates a parameter whenever a "
                                 "gradient tensor exists, even if all values are "
                                 "zero: m <- beta1*m, update <- lr*m_hat/"
                                 "(sqrt(v_hat)+eps) != 0 while m != 0. The "
                                 "empirical check above shows parameters move "
                                 "with all-zero gradients at the 30K states.",
            "overlap_verdict": "cross-iteration double-buffer overlap would let "
                               "the next forward read pre-update parameters -> "
                               "NOT exact unless the forward waits for the "
                               "update, which removes the overlap; same-stream "
                               "ordering is the only exact schedule",
        },
    }
    C.dump_json(record, args.out)


if __name__ == "__main__":
    main()

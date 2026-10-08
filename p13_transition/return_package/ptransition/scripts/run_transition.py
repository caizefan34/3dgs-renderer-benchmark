"""P-TRANSITION: LR-swept local chart comparison built on PRISM-GS v0.1.2.

Place this file next to the existing `prism/` package and run from that bundle root.
It deliberately excludes screen_one_step and full training.
"""
import argparse, hashlib, importlib, json, os, random, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

from prism import gsplat_backend as B
from prism.geometry import covariance, pack_geometry, unpack_geometry, fit_consensus, jacobian


def save(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(obj, indent=2, allow_nan=False))
    t.replace(p)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def parse_grid(s):
    vals = [float(x) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("empty LR grid")
    return vals


def main():
    ap = argparse.ArgumentParser()
    for n in ("gx-path", "scene", "snapshot", "camera-seq", "out"):
        ap.add_argument("--" + n, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-active", type=int, default=512)
    ap.add_argument("--inner-steps", type=int, default=3)
    ap.add_argument("--screen-lrs", default="0.01,0.03,0.1")
    ap.add_argument("--geometry-lrs", default="0.001,0.003,0.01")
    args = ap.parse_args()

    screen_lrs = parse_grid(args.screen_lrs)
    geometry_lrs = parse_grid(args.geometry_lrs)
    out = {
        "status": "STARTED",
        "scope": "P-TRANSITION local chart LR sweep; no full training",
        "scene": args.scene, "iteration": args.iteration, "seed": args.seed,
        "screen_lrs": screen_lrs, "geometry_lrs": geometry_lrs,
        "limitations": [
            "geometry control is the Section-7 9-DoF geometry chart, not original-training Adam",
            "fixed local active set of at most 512 common-visible Gaussians",
            "three local update steps only",
        ],
    }
    save(args.out, out)

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        sys.path.insert(0, args.gx_path)
        G = importlib.import_module("gx_common")
        trainer, ident = G.boot_gmain(args.gpu)
        out["identity"] = G.run_identity(trainer, ident, args.gpu)
        out["fused_loss"] = G.check_fused_lock()
        out["input_sha256"] = {p: sha(p) for p in (args.snapshot, args.camera_seq)}
        out["adapter_source_sha256"] = sha(B.__file__)
        out["gpu_before"] = G.gpu_inventory()

        torch.manual_seed(args.seed)
        np.random.seed(args.seed & 0xFFFFFFFF)

        snap = torch.load(args.snapshot, map_location="cpu", weights_only=False)
        params = {
            "means": snap["xyz"].cuda().float(),
            "quats": snap["rotation"].cuda().float(),
            "log_scales": snap["scaling"].cuda().float(),
            "opacity_logits": snap["opacity"].cuda().float().reshape(-1),
            "sh0": snap["shs"][:, :1].cuda().float(),
            "shN": snap["shs"][:, 1:].cuda().float(),
        }
        degree = int(snap["active_sh_degree"])

        sequence = np.load(args.camera_seq)
        camera_ids = []
        for c in sequence:
            c = int(c)
            if c not in camera_ids:
                camera_ids.append(c)
            if len(camera_ids) == 5:
                break
        if len(camera_ids) < 5:
            raise RuntimeError("Need five distinct TRAINING camera IDs")

        dataset = G.build_dataset(args.scene)
        cameras, targets = [], []
        for c in camera_ids:
            cam, gt = dataset.get_item(c)
            cam = G.camera_dict(cam)
            if tuple(gt.shape) == (3, cam["height"], cam["width"]):
                gt = gt.permute(1, 2, 0)
            if tuple(gt.shape) != (cam["height"], cam["width"], 3):
                raise RuntimeError("Invalid target dimensions")
            cameras.append(cam)
            targets.append(gt.float().contiguous())
        if len({(c["width"], c["height"]) for c in cameras[:3]}) != 1:
            raise RuntimeError("Equal training-view resolution required; do not resize")

        pre_start = sync_time()
        with torch.no_grad():
            states, before = [], []
            for c, t in zip(cameras, targets):
                image, st = B.state(params, c, degree)
                before.append(float(G.fused_loss(image, t)))
                states.append(B.detach_state(st))
            valid = torch.stack([s["mask"][0] for s in states[:3]]).all(0)
            eligible = valid.nonzero().flatten()
            if len(eligible) < 8:
                raise RuntimeError("Insufficient common visibility: INCONCLUSIVE_OBSERVABILITY")
            perm = torch.randperm(len(eligible), device="cuda")
            ids = eligible[perm[: args.max_active]]
            cov = covariance(params["quats"], params["log_scales"])
            pack_stats = {}
            x0 = pack_geometry(params["means"][ids], cov[ids], stats=pack_stats)
            z0 = torch.stack([B.features(s, ids) for s in states[:3]])

        out["camera_ids"] = camera_ids
        out["selection"] = {
            "N": len(params["means"]),
            "common_visible": len(eligible),
            "active": len(ids),
            "ids_sha256": hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest(),
        }
        out["geometry_pack"] = pack_stats
        out["before_per_view_loss"] = before

        views = torch.stack([c["viewmat"] for c in cameras[:3]])
        ks = torch.stack([c["K"] for c in cameras[:3]])
        w, h = cameras[0]["width"], cameras[0]["height"]
        scale = x0.new_tensor([1, 1, .1, .1, .1])

        _, jcheck = jacobian(x0.double(), views.double(), ks.double(), w, h)
        pos = jcheck[:, :, :2, :3].permute(1, 0, 2, 3).reshape(len(ids), 6, 3)
        singular = torch.linalg.svdvals(pos)
        relative = singular[:, -1] / singular[:, 0].clamp_min(1e-20)
        out["position_observability"] = {
            "min_relative_singular": float(relative.min()),
            "median_relative_singular": float(relative.median()),
            "fraction_below_1e_minus_6": float((relative < 1e-6).double().mean()),
        }
        del jcheck, pos, singular
        out["shared_setup_seconds"] = sync_time() - pre_start

        def actual(xx):
            means = params["means"].clone()
            covars = cov.clone()
            m, co = unpack_geometry(xx)
            means[ids] = m
            covars[ids] = co
            return means, covars

        def evaluate(xx):
            means, covars = actual(xx)
            with torch.no_grad():
                return [
                    float(G.fused_loss(B.baseline(params, c, degree, covars, means)[0], t))
                    for c, t in zip(cameras, targets)
                ]

        candidates = [("screen", lr) for lr in screen_lrs] + [
            ("geometry", lr) for lr in geometry_lrs
        ]
        rnd = random.Random(args.seed + 1000003 * args.iteration)
        rnd.shuffle(candidates)
        out["execution_order"] = [{"chart": a, "lr": lr} for a, lr in candidates]
        out["candidates"] = []

        for chart, lr in candidates:
            start = sync_time()
            torch.cuda.reset_peak_memory_stats()
            trace, fit = [], []

            if chart == "screen":
                z = z0.detach().clone().requires_grad_()
                opt = torch.optim.Adam([z], lr=lr)
                for step in range(args.inner_steps):
                    opt.zero_grad()
                    photo = sum(
                        G.fused_loss(B.draw(states[i], ids, z[i]), targets[i])
                        for i in range(3)
                    ) / 3
                    objective = photo + .001 * ((z - z0) / scale).square().mean()
                    objective.backward()
                    opt.step()
                    with torch.no_grad():
                        z.copy_(z0 + (z - z0).clamp(-scale, scale))
                    trace.append({"step": step, "photo_before_step": float(photo.detach())})
                xx, fit = fit_consensus(
                    x0.double(), z.detach().double(), views.double(), ks.double(), w, h,
                    torch.ones_like(z, dtype=torch.float64) / scale.double().square(),
                    steps=3,
                )
                xx = xx.float()
            else:
                xx = x0.detach().clone().requires_grad_()
                opt = torch.optim.Adam([xx], lr=lr)
                for step in range(args.inner_steps):
                    opt.zero_grad()
                    means, covars = actual(xx)
                    photo = sum(
                        G.fused_loss(
                            B.baseline(params, cameras[i], degree, covars, means)[0],
                            targets[i],
                        )
                        for i in range(3)
                    ) / 3
                    photo.backward()
                    opt.step()
                    trace.append({"step": step, "photo_before_step": float(photo.detach())})
                xx = xx.detach()

            optimization_seconds = sync_time() - start
            losses = evaluate(xx)
            with_eval_seconds = sync_time() - start
            train_delta = sum(losses[:3]) / 3 - sum(before[:3]) / 3
            audit_delta = sum(losses[3:]) / 2 - sum(before[3:]) / 2
            all_in = out["shared_setup_seconds"] + with_eval_seconds
            train_gain = -train_delta
            eta = train_gain / all_in if train_gain > 0 else 0.0
            audit_per_train = audit_delta / max(train_gain, 1e-20)

            out["candidates"].append({
                "chart": chart, "lr": lr, "losses": losses, "trace": trace,
                "fit_history": fit,
                "optimization_seconds": optimization_seconds,
                "with_evaluation_seconds": with_eval_seconds,
                "all_in_seconds": all_in,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "train_delta": train_delta, "audit_delta": audit_delta,
                "train_gain": train_gain, "eta": eta,
                "audit_per_train": audit_per_train,
                "full_resolution_backward_views": args.inner_steps * 3,
                "evaluation_forward_views": 5,
            })
            save(args.out, out)

        # Selection rule is preregistered: maximum all-in eta within each fixed grid.
        best = {}
        for chart in ("screen", "geometry"):
            xs = [x for x in out["candidates"] if x["chart"] == chart]
            best[chart] = max(xs, key=lambda x: x["eta"])
        out["best_by_eta"] = {
            k: {
                "lr": v["lr"], "eta": v["eta"], "train_gain": v["train_gain"],
                "audit_per_train": v["audit_per_train"],
                "peak_allocated_bytes": v["peak_allocated_bytes"],
            } for k, v in best.items()
        }
        out["R_eta"] = best["screen"]["eta"] / max(best["geometry"]["eta"], 1e-30)
        out["screen_memory_ratio"] = (
            best["screen"]["peak_allocated_bytes"] /
            max(best["geometry"]["peak_allocated_bytes"], 1)
        )
        out["gpu_after"] = G.gpu_inventory()
        out["status"] = "P_TRANSITION_UNIT_COMPLETE"
        save(args.out, out)

    except Exception:
        out["status"] = "ERROR"
        out["traceback"] = traceback.format_exc()
        save(args.out, out)
        raise


if __name__ == "__main__":
    main()

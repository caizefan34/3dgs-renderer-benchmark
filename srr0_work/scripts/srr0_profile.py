#!/usr/bin/env python
"""SRR0 Gate-4 profiling harness for FasterGSFused preprocess_backward.

Behavior-inert: loads a frozen checkpoint, initializes Adam moments to zero
(exact training_setup allocation), runs native forward+backward on a real
training view, profiles per-kernel CUDA self time with torch.profiler.
NO kernel change, NO training-semantics change, NO checkpoint mutation.
EPIC-05 A100 only. Records remote identity before any experimental command.
"""
import argparse, json, os, sys, time, platform, socket
import torch

sys.path.insert(0, "/root/epic05_p16rb/src/nerficg/src")

def prove_identity():
    return {
        "hostname": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cwd": os.getcwd(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_mem_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else None,
    }

def init_moments(g):
    dev = g.means.device
    g.moments_means = torch.zeros(*g.means.shape, 2, dtype=torch.float32, device=dev)
    g.moments_sh_coefficients_0 = torch.zeros(*g._sh_coefficients_0.shape, 2, dtype=torch.float32, device=dev)
    g.moments_sh_coefficients_rest = torch.zeros(*g._sh_coefficients_rest.shape, 2, dtype=torch.float32, device=dev)
    g.moments_scales = torch.zeros(*g._scales.shape, 2, dtype=torch.float32, device=dev)
    g.moments_rotations = torch.zeros(*g._rotations.shape, 2, dtype=torch.float32, device=dev)
    g.moments_opacities = torch.zeros(*g._opacities.shape, 2, dtype=torch.float32, device=dev)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset-root", default="/root/epic05_p16rb/common_data/mipnerf360")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--view-index", type=int, default=0)
    ap.add_argument("--adam-step", type=int, default=29000)
    ap.add_argument("--lr-means", type=float, default=1.6e-6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--view-ids", default=None)
    ap.add_argument("--timing-iters", type=int, default=10)
    args = ap.parse_args()

    ident = prove_identity()
    print("IDENTITY", json.dumps(ident), flush=True)
    if "A100" not in (ident["gpu"] or ""):
        print("ERROR: not an A100; remote guard STOP_REMOTE_GUARD", flush=True); sys.exit(2)

    import Framework
    from Implementations import Methods as MI
    from Implementations import Datasets as DI

    cfg_map = {
        "room": "p16_fastergsfused_room_1920.yaml",
        "garden": "p16_fastergsfused_garden_1920.yaml",
        "bicycle": "p16_fastergsfused_bicycle_1920.yaml",
    }
    cfg_path = os.path.join("/root/srr0_configs", cfg_map[args.scene])
    Framework.setup(config_path=cfg_path, require_custom_config=True)

    dataset = DI.get_dataset(dataset_type=Framework.config.GLOBAL.DATASET_TYPE,
                            path=os.path.join(args.dataset_root, args.scene))
    dataset.train()
    view = dataset.train()[args.view_index]

    method = Framework.config.GLOBAL.METHOD_TYPE
    model = MI.get_model(method=method, checkpoint=args.ckpt)
    g = model.gaussians
    init_moments(g)
    g.lr_means = args.lr_means
    renderer = MI.get_renderer(method=method, model=model)

    print("LOAD", json.dumps({
        "n_gaussians": int(g.means.shape[0]),
        "active_sh_bases": int(g.active_sh_bases),
        "max_sh_degree": int(g.max_sh_degree),
        "ckpt": args.ckpt,
        "view": args.view_index,
        "img": [int(view.camera.width), int(view.camera.height)],
    }), flush=True)

    autograd_dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device='cuda'))

    # Use the trainer's loss for fidelity (config TRAINING.LOSS)
    from Methods.FasterGSFused.Loss import FasterGSFusedLoss
    loss_fn = FasterGSFusedLoss(loss_config=Framework.config.TRAINING.LOSS)

    def step():
        model.train()
        bg = view.camera.background_color
        image, ad = renderer.render_image_training(
            view=view, update_densification_info=False, bg_color=bg,
            adam_step_count=args.adam_step, autograd_dummy=autograd_dummy,
        )
        rgb_gt = view.rgb
        loss = loss_fn(image, rgb_gt) + 0.0 * ad
        loss.backward()
        return loss.item()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.iters):
            step()
        torch.cuda.synchronize()

    # Multi-view wall-clock timing (median over N views) for E2E ceiling
    import statistics
    view_times = []
    n_views = args.views
    view_ids = [int(x) for x in args.view_ids.split(",")] if args.view_ids else list(range(min(n_views, 8)))
    for vid in view_ids:
        v2 = dataset.train()[vid]
        def step_v(vv):
            model.train()
            bg = vv.camera.background_color
            image, ad = renderer.render_image_training(
                view=vv, update_densification_info=False, bg_color=bg,
                adam_step_count=args.adam_step, autograd_dummy=autograd_dummy,
            )
            loss = loss_fn(image, vv.rgb) + 0.0 * ad
            loss.backward()
            return loss.item()
        for _ in range(args.warmup):
            step_v(v2)
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.timing_iters):
            t0 = time.perf_counter()
            step_v(v2)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        view_times.append({"view": vid, "median_ms": float(statistics.median(ts)),
                           "min_ms": float(min(ts)), "max_ms": float(max(ts))})
    wall_median_ms = float(statistics.median([vt["median_ms"] for vt in view_times]))
    print("WALL", json.dumps({"wall_median_ms": wall_median_ms, "views": view_times}), flush=True)

    ka = prof.key_averages()
    kernel_rows = []
    total_cuda_us = 0.0
    for ev in ka:
        if ev.self_device_time_total > 0:
            kernel_rows.append({"name": ev.key, "self_cuda_us": float(ev.self_device_time_total),
                                "count": int(ev.count)})
            total_cuda_us += float(ev.self_device_time_total)
    kernel_rows.sort(key=lambda r: -r["self_cuda_us"])

    out = {
        "identity": ident, "scene": args.scene, "ckpt": args.ckpt,
        "n_gaussians": int(g.means.shape[0]), "active_sh_bases": int(g.active_sh_bases),
        "view_index": args.view_index, "iters": args.iters, "warmup": args.warmup,
        "total_cuda_self_us_per_iter": total_cuda_us / max(args.iters, 1),
        "wall_median_ms": wall_median_ms,
        "wall_views": view_times,
        "kernels": kernel_rows, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    outpath = args.out or f"/root/srr0_{args.scene}_profile.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE", outpath, flush=True)
    for r in kernel_rows[:14]:
        print(f"KERNEL {r['self_cuda_us']/max(args.iters,1):.1f}us x{r['count']} {r['name']}", flush=True)

if __name__ == "__main__":
    main()

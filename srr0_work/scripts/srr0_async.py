#!/usr/bin/env python
"""SRR0 Gate-4 async-throughput harness. Behavior-inert.
Measures steady-state per-iteration cost WITHOUT per-iter sync to determine
whether the iteration is CPU-bound or GPU-bound. NO training step, NO checkpoint
mutation, NO kernel change. EPIC-05 A100 only. Records remote identity.
"""
import argparse, json, os, sys, time, platform, socket, statistics
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
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--sync-iters", type=int, default=20)
    ap.add_argument("--async-iters", type=int, default=60)
    ap.add_argument("--view-index", type=int, default=0)
    ap.add_argument("--adam-step", type=int, default=29000)
    ap.add_argument("--lr-means", type=float, default=1.6e-6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ident = prove_identity()
    print("IDENTITY", json.dumps(ident), flush=True)
    if "A100" not in (ident["gpu"] or ""):
        print("ERROR: not an A100; STOP_REMOTE_GUARD", flush=True); sys.exit(2)

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

    # (A) per-iter sync latency (reproduces wall)
    sync_lat = []
    for _ in range(args.sync_iters):
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        sync_lat.append((time.perf_counter() - t0) * 1e3)

    # (B) async steady-state throughput (no intermediate sync)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.async_iters):
        step()
    torch.cuda.synchronize()
    async_total = (time.perf_counter() - t0) * 1e3
    async_per_iter = async_total / args.async_iters

    # (C) GPU-only lower bound via events around a single step (many iters, one sync)
    torch.cuda.synchronize()
    start_ev = [torch.cuda.Event(enable_timing=True) for _ in range(args.async_iters)]
    end_ev = [torch.cuda.Event(enable_timing=True) for _ in range(args.async_iters)]
    torch.cuda.synchronize()
    for i in range(args.async_iters):
        start_ev[i].record()
        step()
        end_ev[i].record()
    torch.cuda.synchronize()
    gpu_only = [s.elapsed_time(e) for s, e in zip(start_ev, end_ev)]

    out = {
        "identity": ident, "scene": args.scene, "ckpt": args.ckpt,
        "n_gaussians": int(g.means.shape[0]), "view_index": args.view_index,
        "sync_latency_median_ms": float(statistics.median(sync_lat)),
        "sync_latency_min_ms": float(min(sync_lat)),
        "async_per_iter_ms": float(async_per_iter),
        "async_total_ms": float(async_total),
        "async_iters": args.async_iters,
        "gpu_only_median_ms": float(statistics.median(gpu_only)),
        "gpu_only_min_ms": float(min(gpu_only)),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    outpath = args.out or f"/root/srr0_{args.scene}_async.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print("ASYNC", json.dumps(out), flush=True)
    print("WROTE", outpath, flush=True)

if __name__ == "__main__":
    main()

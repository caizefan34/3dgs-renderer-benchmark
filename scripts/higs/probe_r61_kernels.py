#!/usr/bin/env python
"""Round-61 kernel profiler: masked-adam 720p eg garden, top kernels by CUDA time."""
import argparse, importlib.util, json, os, sys
import torch
import lpips

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmark"))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BENCH = os.path.join(_REPO, "benchmark", "run_higs_train_benchmark.py")
_spec = importlib.util.spec_from_file_location("run_higs_train_benchmark", _BENCH)
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}")

    scene_dir = os.path.join(args.base_dir, args.scene)
    params0 = B.load_ply_scene(os.path.join(scene_dir, "point_cloud.ply"), device)
    print(f"scene={args.scene} n_gaussians={params0[0].shape[0]}")
    viewmats, Ks, train_idx, eval_idx = B.load_cameras(
        scene_dir, args.width, args.height, 4, 3, device,
    )
    with open(os.path.join(scene_dir, "eval_cameras.json")) as f:
        cams = json.load(f)
    refs_train = B.load_reference(
        os.path.join(scene_dir, "eval_images"),
        [cams[i] for i in train_idx], args.width, args.height, device,
    )
    refs_eval = B.load_reference(
        os.path.join(scene_dir, "eval_images"),
        [cams[i] for i in eval_idx], args.width, args.height, device,
    )

    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    kw = dict(
        backend="higs_dynamic_ts", params0=params0, viewmats=viewmats, Ks=Ks,
        train_idx=train_idx, refs_train=refs_train, eval_idx=eval_idx,
        refs_eval=refs_eval, width=args.width, height=args.height,
        steps=args.steps, seed=0, device=device,
        densify_every=5, densify_threshold=0.005, prune_threshold=0.01,
        lpips_model=lpips_model, tile_sampling_ratio=0.35,
        sampling_mode="error_guided", error_alpha=1.0,
        error_refresh_every=25, error_lambda=0.7, eval_every=60,
        lr_decay=0.1, densify_window=1500,
        lpips_loss_weight=0.1, lpips_loss_every=25, lpips_full_res=True,
        anchor_densify=True, anchor_densify_every=2,
        cull_interval=1, masked_adam=True,
    )
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        r = B.run_backend(**kw)
    print(json.dumps({k: r[k] for k in (
        "fwd_ms", "bwd_ms", "total_ms", "train_ms", "culling_ratio",
        "n_visible_avg", "final_n", "psnr", "n_mask_pruned",
    )}, indent=1))
    print("=== TOP 30 KERNELS (self CUDA time) ===")
    evs = prof.key_averages()
    tot = sum(e.self_device_time_total for e in evs) or 1
    evs = sorted(evs, key=lambda e: -e.self_device_time_total)[:30]
    for e in evs:
        frac = e.self_device_time_total / tot * 100
        print(f"{e.self_device_time_total/1000:9.2f} ms  {frac:5.1f}%  count={e.count:6d}  {e.key[:110]}")
    print(f"TOTAL self CUDA {tot/1000:.1f} ms over {args.steps} steps")


if __name__ == "__main__":
    main()
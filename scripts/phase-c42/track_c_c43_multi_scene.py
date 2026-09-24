#!/usr/bin/env python3
"""
Track C: C43 Tile-size validation across scenes.
Usage: python3 track_c_c43_multi_scene.py --scene bicycle
"""
import json, math, sys, time, argparse
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "epic05" / "phase7"))
from gsplat import rasterization, fully_fused_projection, isect_tiles
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint

DEVICE = "cuda"; PACKED = True; EPS2D = 0.1; RADIUS_CLIP = 0.0; SEED = 42


def d_ssim_loss(pred, target, window_size=11, sigma=1.5):
    if pred.ndim == 3: pred = pred.unsqueeze(0).permute(0,3,1,2); target = target.unsqueeze(0).permute(0,3,1,2)
    coords = torch.arange(window_size, device=pred.device, dtype=pred.dtype) - window_size//2
    k1d = torch.exp(-(coords**2)/(2*sigma**2)); k1d = k1d/k1d.sum()
    kernel = (k1d[:,None]*k1d[None,:]).expand(pred.shape[1],1,window_size,window_size).contiguous()
    C1, C2 = 0.01**2, 0.03**2
    def blur(x): return F.conv2d(x, kernel, padding=window_size//2, groups=pred.shape[1])
    mu_p, mu_t = blur(pred), blur(target)
    ssim = ((2*mu_p*mu_t+C1)*(2*(blur(pred*target)-mu_p*mu_t)+C2))/((mu_p**2+mu_t**2+C1)*(blur(pred**2)-mu_p**2+blur(target**2)-mu_t**2+C2))
    return 1.0 - ssim.mean()


def render_with_tile(model, cam, tile_size):
    data = model.forward()
    r, _, _ = rasterization(means=data["xyz"],quats=data["rotations"],scales=data["scales"],
        opacities=data["opacity"],colors=data["shs"],viewmats=cam.viewmatrix.unsqueeze(0),
        Ks=cam.K.unsqueeze(0),width=cam.image_width,height=cam.image_height,
        tile_size=tile_size,packed=PACKED,sh_degree=model.sh_degree,
        radius_clip=RADIUS_CLIP,eps2d=EPS2D,render_mode="RGB")
    return r[0].clamp(0,1)


def benchmark_tile(model, dataset, cam_indices, tile_size, label):
    print(f"\n  [{label}] tile_size={tile_size}")
    # Warmup
    for ci in cam_indices[:3]:
        with torch.no_grad(): _ = render_with_tile(model, dataset.get_camera(ci), tile_size)
    torch.cuda.synchronize()

    render_times, psnrs, ssims, n_isects_list = [], [], [], []
    for ci in cam_indices:
        cam = dataset.get_camera(ci); gt = dataset.get_gt_image(ci)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(): pred = render_with_tile(model, cam, tile_size)
        torch.cuda.synchronize()
        render_times.append((time.perf_counter()-t0)*1000)
        mse = float(((pred-gt)**2).mean())
        psnrs.append(10*math.log10(1.0/max(mse,1e-10)))
        ssims.append(1.0-float(d_ssim_loss(pred,gt)))

        # Count intersections
        data = model.forward()
        radii, means2d, depths, _, _ = fully_fused_projection(
            means=data["xyz"], covars=None, quats=data["rotations"], scales=data["scales"],
            viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
            width=cam.image_width, height=cam.image_height,
            radius_clip=RADIUS_CLIP, packed=False, eps2d=EPS2D)
        tw = (cam.image_width + tile_size - 1) // tile_size
        th = (cam.image_height + tile_size - 1) // tile_size
        _, _, flatten_ids = isect_tiles(means2d, radii, depths, tile_size, tw, th, sort=True, packed=False)
        n_isects_list.append(len(flatten_ids))

    print(f"    mean render: {np.mean(render_times):.2f} ms  PSNR: {np.mean(psnrs):.2f}  SSIM: {np.mean(ssims):.4f}")
    print(f"    mean n_isects: {int(np.mean(n_isects_list)):,}")
    return {
        "label": label, "tile_size": tile_size,
        "mean_render_ms": float(np.mean(render_times)),
        "std_render_ms": float(np.std(render_times)),
        "mean_psnr": float(np.mean(psnrs)),
        "mean_ssim": float(np.mean(ssims)),
        "mean_n_isects": int(np.mean(n_isects_list)),
        "per_camera": [{"cam": ci, "render_ms": rt, "psnr": p, "ssim": s, "n_isects": ni}
                       for ci, rt, p, s, ni in zip(cam_indices, render_times, psnrs, ssims, n_isects_list)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="room")
    args = parser.parse_args()
    scene = args.scene

    print(f"{'='*72}")
    print(f"Track C: C43 Tile-size Validation ({scene})")
    print(f"{'='*72}")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"  GPU: {gpu_name}  Scene: {scene}")

    torch.manual_seed(SEED); np.random.seed(SEED)
    repo_root = Path(__file__).resolve().parent.parent.parent
    dataset = GTDataset(scene=scene, repo_root=repo_root, resolution="1080p", device=DEVICE)
    print(f"  Cameras: {len(dataset)}")
    n_cams = len(dataset)
    eval_cams = list(range(0, n_cams, max(1, n_cams//13)))[:13]

    # Load SfM init
    sfm_data = load_initial_checkpoint(scene, repo_root, device=DEVICE)
    model = GaussianModel(num_points=sfm_data["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=DEVICE)
    model.init_from_sfm(xyz=sfm_data["xyz"],
        opacity_logit=torch.logit(torch.full((sfm_data["xyz"].shape[0],1),0.1,device=DEVICE)),
        scales_log=sfm_data.get("scales"), rotations_raw=sfm_data.get("rotations"), shs=sfm_data.get("shs"))
    model.set_sh_degree(3)
    print(f"  SfM points: {model.xyz.shape[0]:,}")

    # Benchmark tile16, tile32, tile24
    r16 = benchmark_tile(model, dataset, eval_cams, 16, "tile16")
    r32 = benchmark_tile(model, dataset, eval_cams, 32, "tile32")
    r24 = benchmark_tile(model, dataset, eval_cams, 24, "tile24")

    speedup_32 = (1 - r32["mean_render_ms"]/r16["mean_render_ms"]) * 100
    speedup_24 = (1 - r24["mean_render_ms"]/r16["mean_render_ms"]) * 100
    psnr_diff_32 = r32["mean_psnr"] - r16["mean_psnr"]
    psnr_diff_24 = r24["mean_psnr"] - r16["mean_psnr"]

    print(f"\n  tile16 vs tile32: speedup={speedup_32:+.1f}%  dPSNR={psnr_diff_32:+.2f}")
    print(f"  tile16 vs tile24: speedup={speedup_24:+.1f}%  dPSNR={psnr_diff_24:+.2f}")

    output = {
        "experiment": f"Track C C43 Tile-size Validation ({scene})",
        "hardware": {"gpu": gpu_name, "sms": gpu_props.multi_processor_count},
        "config": {"scene": scene, "n_sfm_points": model.xyz.shape[0], "eval_cameras": eval_cams},
        "results": {"tile16": r16, "tile32": r32, "tile24": r24},
        "analysis": {"speedup_32_vs_16": speedup_32, "speedup_24_vs_16": speedup_24,
                     "dpsnr_32": psnr_diff_32, "dpsnr_24": psnr_diff_24},
    }
    save_path = repo_root / "results" / "a100" / "phase-c42" / f"track_c_c43_{scene}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Data saved to {save_path}")

if __name__ == "__main__":
    main()

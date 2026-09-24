#!/usr/bin/env python3
"""
Track A: Multi-scene validation (baseline vs scale=0.75, 5K iters).
Usage: python3 track_a_multi_scene.py --scene bicycle
       python3 track_a_multi_scene.py --scene garden
"""
import json, math, sys, time, argparse
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "epic05" / "phase7"))
from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint

DEVICE = "cuda"; TILE_SIZE = 16; PACKED = True; EPS2D = 0.1; RADIUS_CLIP = 0.0
LAMBDA_DSSIM = 0.2; SEED = 42; NUM_ITERS = 5000; EVAL_INTERVAL = 500
DENSIFICATION_INTERVAL = 100; DENSIFICATION_GRAD_THRESHOLD = 0.0002
DENSIFICATION_START = 500; DENSIFICATION_END = 15000
CLONE_MAX_SCREEN_SIZE = 100.0; SPLIT_MAX_SCREEN_SIZE = 100.0
PRUNE_INTERVAL = 100; PRUNE_OPACITY_THRESHOLD = 0.005; PRUNE_START = 500
RESET_OPACITY_INTERVAL = 3000; SH_DEGREE_INTERVAL = 1000; MAX_SH_DEGREE = 3


def d_ssim_loss(pred, target, scale=1.0, window_size=11, sigma=1.5):
    if pred.ndim == 3: pred = pred.unsqueeze(0).permute(0,3,1,2); target = target.unsqueeze(0).permute(0,3,1,2)
    if scale < 1.0:
        pred = F.interpolate(pred, scale_factor=scale, mode="area", recompute_scale_factor=False)
        target = F.interpolate(target, scale_factor=scale, mode="area", recompute_scale_factor=False)
    coords = torch.arange(window_size, device=pred.device, dtype=pred.dtype) - window_size//2
    k1d = torch.exp(-(coords**2)/(2*sigma**2)); k1d = k1d/k1d.sum()
    kernel = (k1d[:,None]*k1d[None,:]).expand(pred.shape[1],1,window_size,window_size).contiguous()
    C1, C2 = 0.01**2, 0.03**2
    def blur(x): return F.conv2d(x, kernel, padding=window_size//2, groups=pred.shape[1])
    mu_p, mu_t = blur(pred), blur(target)
    ssim = ((2*mu_p*mu_t+C1)*(2*(blur(pred*target)-mu_p*mu_t)+C2))/((mu_p**2+mu_t**2+C1)*(blur(pred**2)-mu_p**2+blur(target**2)-mu_t**2+C2))
    return 1.0 - ssim.mean()

def compute_loss(pred, gt, scale):
    l1 = F.l1_loss(pred, gt); dsim = d_ssim_loss(pred, gt, scale=scale)
    return (1.0-LAMBDA_DSSIM)*l1 + LAMBDA_DSSIM*dsim, l1, dsim

def make_optimizer(model, sls):
    return torch.optim.Adam([
        {"params":[model.xyz],"lr":1.6e-4*sls,"eps":1e-15,"betas":(0.9,0.999)},
        {"params":[model.rotations],"lr":1e-3,"eps":1e-15,"betas":(0.9,0.999)},
        {"params":[model.scales],"lr":5e-3,"eps":1e-15,"betas":(0.9,0.999)},
        {"params":[model.opacity],"lr":5e-2,"eps":1e-15,"betas":(0.9,0.999)},
        {"params":[model.shs],"lr":2.5e-3,"eps":1e-15,"betas":(0.9,0.999)},
    ])

def render(model, cam):
    data = model.forward()
    r, _, _ = rasterization(means=data["xyz"],quats=data["rotations"],scales=data["scales"],
        opacities=data["opacity"],colors=data["shs"],viewmats=cam.viewmatrix.unsqueeze(0),
        Ks=cam.K.unsqueeze(0),width=cam.image_width,height=cam.image_height,
        tile_size=TILE_SIZE,packed=PACKED,sh_degree=model.sh_degree,
        radius_clip=RADIUS_CLIP,eps2d=EPS2D,render_mode="RGB")
    return r[0].clamp(0,1)

def evaluate(model, dataset, cam_indices):
    psnrs, ssims = [], []
    for ci in cam_indices:
        cam = dataset.get_camera(ci); gt = dataset.get_gt_image(ci)
        with torch.no_grad():
            pred = render(model, cam)
            mse = float(((pred-gt)**2).mean())
            psnrs.append(10*math.log10(1.0/max(mse,1e-10)))
            ssims.append(1.0-float(d_ssim_loss(pred,gt,scale=1.0)))
    return float(np.mean(psnrs)), float(np.mean(ssims))

def train_variant(dataset, sfm_data, scale, name):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = GaussianModel(num_points=sfm_data["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=DEVICE)
    model.init_from_sfm(xyz=sfm_data["xyz"],
        opacity_logit=torch.logit(torch.full((sfm_data["xyz"].shape[0],1),0.1,device=DEVICE)),
        scales_log=sfm_data.get("scales"),rotations_raw=sfm_data.get("rotations"),shs=sfm_data.get("shs"))
    sls = float(sfm_data["xyz"].norm(dim=-1).max().item())
    optimizer = make_optimizer(model, sls)
    num_cameras = len(dataset)
    n_cams = num_cameras
    eval_cams = list(range(0, n_cams, max(1, n_cams//13)))[:13]

    results = {"variant": name, "scale": scale, "eval_points": [], "timing": {}}
    cum_cloned=0; cum_split=0; cum_pruned=0
    iter_times = []; t0_total = time.perf_counter()

    for iteration in range(1, NUM_ITERS+1):
        cam_idx = (iteration-1) % num_cameras
        cam = dataset.get_camera(cam_idx); gt = dataset.get_gt_image(cam_idx)
        new_deg = min(MAX_SH_DEGREE, iteration // SH_DEGREE_INTERVAL)
        if new_deg != model.sh_degree and new_deg <= MAX_SH_DEGREE:
            model.set_sh_degree(new_deg); optimizer = make_optimizer(model, sls)
        t0 = time.perf_counter()
        pred = render(model, cam)
        loss, l1_val, dsim_val = compute_loss(pred, gt, scale)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        model.accumulate_positional_gradient()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        denf = {"cloned":0,"split":0,"removed":0}
        if iteration >= DENSIFICATION_START and iteration < DENSIFICATION_END and iteration % DENSIFICATION_INTERVAL == 0:
            denf = model.densification(grad_threshold=DENSIFICATION_GRAD_THRESHOLD,
                clone_max_screen_size=CLONE_MAX_SCREEN_SIZE, split_max_screen_size=SPLIT_MAX_SCREEN_SIZE)
        prune_count = 0
        if iteration >= PRUNE_START and iteration % PRUNE_INTERVAL == 0:
            prune_count = model.prune_and_reset(opacity_threshold=PRUNE_OPACITY_THRESHOLD,
                reset_interval=RESET_OPACITY_INTERVAL, current_step=iteration)
        if denf["cloned"]+denf["split"]+prune_count > 0:
            cum_cloned+=denf["cloned"]; cum_split+=denf["split"]; cum_pruned+=prune_count
            optimizer = make_optimizer(model, sls)
        torch.cuda.synchronize()
        iter_times.append((time.perf_counter()-t0)*1000)
        if iteration % EVAL_INTERVAL == 0:
            psnr, ssim = evaluate(model, dataset, eval_cams)
            results["eval_points"].append({"iter": iteration, "psnr": psnr, "ssim": ssim,
                "n_gaussians": model.xyz.shape[0], "mean_iter_ms": float(np.mean(iter_times[-EVAL_INTERVAL:]))})
            print(f"  [{name}@{iteration}] PSNR={psnr:.2f}  SSIM={ssim:.4f}  GS={model.xyz.shape[0]:,}  "
                f"iter={np.mean(iter_times[-EVAL_INTERVAL:]):.1f}ms")

    total = time.perf_counter()-t0_total
    results["timing"] = {"total_wall_s": total, "mean_iter_ms": float(np.mean(iter_times))}
    results["cumulative_topology"] = {"cloned": cum_cloned, "split": cum_split, "pruned": cum_pruned}
    print(f"  {name}: {total:.1f}s  iter={np.mean(iter_times):.2f}ms  clone={cum_cloned:,} split={cum_split:,} prune={cum_pruned:,}")
    del model; torch.cuda.empty_cache()
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="bicycle")
    args = parser.parse_args()
    scene = args.scene

    print(f"{'='*72}")
    print(f"Track A: Multi-scene Validation ({scene})")
    print(f"{'='*72}")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  GPU: {gpu_name}  Scene: {scene}")

    repo_root = Path(__file__).resolve().parent.parent.parent
    dataset = GTDataset(scene=scene, repo_root=repo_root, resolution="1080p", device=DEVICE)
    print(f"  Cameras: {len(dataset)}")
    sfm_data = load_initial_checkpoint(scene, repo_root, device=DEVICE)
    print(f"  SfM points: {sfm_data['xyz'].shape[0]:,}")

    ra = train_variant(dataset, sfm_data, scale=1.0, name="A_baseline")
    rb = train_variant(dataset, sfm_data, scale=0.75, name="B_scale0.75")

    fa, fb = ra["eval_points"][-1], rb["eval_points"][-1]
    d_psnr = fb["psnr"]-fa["psnr"]; d_ssim = fb["ssim"]-fa["ssim"]
    d_gs = (fb["n_gaussians"]-fa["n_gaussians"])/fa["n_gaussians"]*100
    speedup = (1-rb["timing"]["mean_iter_ms"]/ra["timing"]["mean_iter_ms"])*100
    print(f"\n  FINAL: dPSNR={d_psnr:+.2f}  dSSIM={d_ssim:+.4f}  dGS={d_gs:+.1f}%  speedup={speedup:+.1f}%")

    output = {
        "experiment": f"Track A Multi-scene ({scene})",
        "hardware": {"gpu": gpu_name},
        "config": {"seed": SEED, "num_iters": NUM_ITERS, "scene": scene},
        "variant_A_baseline": ra, "variant_B_scale0.75": rb,
        "analysis": {"d_psnr": d_psnr, "d_ssim": d_ssim, "d_gs_pct": d_gs, "speedup_pct": speedup},
    }
    save_path = repo_root / "results" / "a100" / "phase-c42" / f"track_a_multiscene_{scene}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Data saved to {save_path}")

if __name__ == "__main__":
    main()

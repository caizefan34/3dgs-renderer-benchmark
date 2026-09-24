#!/usr/bin/env python3
"""Quick P2 evaluation from saved checkpoints."""
import json, math, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "epic05" / "phase7"))
from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset

DEVICE = "cuda"
TILE_SIZE = 16; PACKED = True; EPS2D = 0.1; RADIUS_CLIP = 0.0

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

def render(model, cam):
    data = model.forward()
    r, _, _ = rasterization(means=data["xyz"],quats=data["rotations"],scales=data["scales"],
        opacities=data["opacity"],colors=data["shs"],viewmats=cam.viewmatrix.unsqueeze(0),
        Ks=cam.K.unsqueeze(0),width=cam.image_width,height=cam.image_height,
        tile_size=TILE_SIZE,packed=PACKED,sh_degree=model.sh_degree,
        radius_clip=RADIUS_CLIP,eps2d=EPS2D,render_mode="RGB")
    return r[0].clamp(0,1)

def evaluate(model, dataset, cam_indices, label):
    print(f"  Evaluating {label} on {len(cam_indices)} cameras...")
    psnrs, ssims = [], []
    t0 = time.perf_counter()
    for i, ci in enumerate(cam_indices):
        cam = dataset.get_camera(ci); gt = dataset.get_gt_image(ci)
        with torch.no_grad():
            pred = render(model, cam)
            mse = float(((pred-gt)**2).mean())
            psnrs.append(10*math.log10(1.0/max(mse,1e-10)))
            ssims.append(1.0-float(d_ssim_loss(pred,gt)))
        if (i+1) % 50 == 0:
            print(f"    {i+1}/{len(cam_indices)}...")
    dt = time.perf_counter() - t0
    print(f"  Done in {dt:.1f}s")
    return float(np.mean(psnrs)), float(np.mean(ssims))

repo_root = Path(__file__).resolve().parent.parent.parent
ckpt_dir = repo_root / "results" / "a100" / "phase-c42" / "p2_checkpoints"
dataset = GTDataset(scene="room", repo_root=repo_root, resolution="1080p", device=DEVICE)
all_cams = list(range(311))

# Evaluate A (baseline, iter 30K)
print("Loading A_baseline_iter30000.pt...")
ckpt_a = torch.load(ckpt_dir / "A_baseline_iter30000.pt", map_location=DEVICE, weights_only=False)
model_a = GaussianModel.from_checkpoint_state(ckpt_a, device=DEVICE)
model_a.set_sh_degree(3)
print(f"  GS={model_a.xyz.shape[0]:,}")
psnr_a, ssim_a = evaluate(model_a, dataset, all_cams, "A (baseline)")
gs_a = model_a.xyz.shape[0]
del model_a; torch.cuda.empty_cache()

# Evaluate B (scale=0.75, iter 30K)
print("\nLoading B_downsampled_0.75_iter30000.pt...")
ckpt_b = torch.load(ckpt_dir / "B_downsampled_0.75_iter30000.pt", map_location=DEVICE, weights_only=False)
model_b = GaussianModel.from_checkpoint_state(ckpt_b, device=DEVICE)
model_b.set_sh_degree(3)
print(f"  GS={model_b.xyz.shape[0]:,}")
psnr_b, ssim_b = evaluate(model_b, dataset, all_cams, "B (scale=0.75)")
gs_b = model_b.xyz.shape[0]
del model_b; torch.cuda.empty_cache()

# Compare
d_psnr = psnr_b - psnr_a
d_ssim = ssim_b - ssim_a
d_gs = (gs_b - gs_a) / gs_a * 100

print(f"\n{'='*60}")
print(f"P2 FINAL EVAL (311 cams, iter 30K)")
print(f"{'='*60}")
print(f"  PSNR:  A={psnr_a:.2f}  B={psnr_b:.2f}  dPSNR={d_psnr:+.2f} dB  (gate: > -0.2)")
print(f"  SSIM:  A={ssim_a:.4f}  B={ssim_b:.4f}  dSSIM={d_ssim:+.4f}  (gate: > -0.005)")
print(f"  GS:    A={gs_a:,}  B={gs_b:,}  dGS={d_gs:+.1f}%  (gate: < 10%)")

psnr_pass = d_psnr > -0.2
ssim_pass = d_ssim > -0.005
gs_pass = abs(d_gs) < 10
# Speedup: we need timing data, estimate from P1
print(f"\n  DECISION GATES:")
print(f"    PSNR drop < 0.2 dB:  {'PASS' if psnr_pass else 'FAIL'} ({d_psnr:+.2f})")
print(f"    SSIM drop < 0.005:   {'PASS' if ssim_pass else 'FAIL'} ({d_ssim:+.4f})")
print(f"    GS diff < 10%:       {'PASS' if gs_pass else 'FAIL'} ({d_gs:+.1f}%)")
print(f"    Speedup > 30%:       (from P1: +33.6%, same scale=0.75)")

if psnr_pass and ssim_pass and gs_pass:
    decision = "PASS -- all gates met at 30K. Scale=0.75 validated for full training."
else:
    failed = [g for g,p in [("PSNR",psnr_pass),("SSIM",ssim_pass),("GS",gs_pass)] if not p]
    decision = f"FAIL -- gates not met: {', '.join(failed)}"
print(f"\n  DECISION: {decision}")

output = {
    "experiment": "C42 P2 Training Validation (30K, scale=0.75) - Final Eval",
    "config": {"seed": 42, "num_iters": 30000, "scene": "room"},
    "variant_A_baseline": {"psnr": psnr_a, "ssim": ssim_a, "n_gaussians": gs_a},
    "variant_B_downsampled_0.75": {"psnr": psnr_b, "ssim": ssim_b, "n_gaussians": gs_b},
    "comparison": {"d_psnr": d_psnr, "d_ssim": d_ssim, "d_gs_pct": d_gs,
                   "psnr_pass": psnr_pass, "ssim_pass": ssim_pass, "gs_pass": gs_pass,
                   "decision": decision},
}
save_path = repo_root / "results" / "a100" / "phase-c42" / "c42_p2_training_validation_30k.json"
with open(save_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n  Data saved to {save_path}")

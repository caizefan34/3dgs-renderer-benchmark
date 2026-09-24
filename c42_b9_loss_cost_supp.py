#!/usr/bin/env python3
"""Supplementary loss-cost benchmark for scales 0.50 and 0.625."""
import torch, torch.nn.functional as F, numpy as np, json, os, sys, hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
REFERENCE_V1_DIR = REPO_ROOT / "baseline" / "reference_v1"
sys.path.insert(0, str(REFERENCE_V1_DIR))

class SepSSIM:
    def __init__(self, window_size=11, sigma=1.5, device="cuda"):
        self.C1 = (0.01) ** 2; self.C2 = (0.03) ** 2
        coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); k1d = k1d / k1d.sum()
        self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
        self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0, 1, 3, 2).contiguous()
        self.padding = window_size // 2
    def __call__(self, pred, target):
        if pred.ndim == 3: pred = pred.unsqueeze(0).permute(0, 3, 1, 2); target = target.unsqueeze(0).permute(0, 3, 1, 2)
        stacked = torch.cat([pred, target, pred**2, target**2, pred*target], dim=1)
        b = F.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=15)
        b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
        mu_p, mu_t = b[:, 0:3], b[:, 3:6]; bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
        mu_p2, mu_t2, mu_pt = mu_p**2, mu_t**2, mu_p*mu_t
        sp2, st2, spt = bp2-mu_p2, bt2-mu_t2, bpt-mu_pt
        ssim_map = (2*mu_pt+self.C1)*(2*spt+self.C2)/((mu_p2+mu_t2+self.C1)*(sp2+st2+self.C2))
        return 1.0 - ssim_map.mean()

def d_ssim_ds(ssim_fn, pred, target, scale):
    if pred.ndim == 3: pred = pred.unsqueeze(0).permute(0, 3, 1, 2); target = target.unsqueeze(0).permute(0, 3, 1, 2)
    if scale < 1.0:
        pred = F.interpolate(pred, scale_factor=scale, mode="area", recompute_scale_factor=False)
        target = F.interpolate(target, scale_factor=scale, mode="area", recompute_scale_factor=False)
    stacked = torch.cat([pred, target, pred**2, target**2, pred*target], dim=1)
    b = F.conv2d(stacked, ssim_fn.k_h, padding=(0, ssim_fn.padding), groups=15)
    b = F.conv2d(b, ssim_fn.k_v, padding=(ssim_fn.padding, 0), groups=15)
    mu_p, mu_t = b[:, 0:3], b[:, 3:6]; bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
    mu_p2, mu_t2, mu_pt = mu_p**2, mu_t**2, mu_p*mu_t
    sp2, st2, spt = bp2-mu_p2, bt2-mu_t2, bpt-mu_pt
    ssim_map = (2*mu_pt+ssim_fn.C1)*(2*spt+ssim_fn.C2)/((mu_p2+mu_t2+ssim_fn.C1)*(sp2+st2+ssim_fn.C2))
    return 1.0 - ssim_map.mean()

LAMBDA = 0.2
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
ssim_fn = SepSSIM(device="cuda")
H, W = 1080, 1920
torch.manual_seed(42)
pred_base = torch.rand(H, W, 3, device="cuda")
target = torch.rand(H, W, 3, device="cuda")

results = {}
for scale in [0.50, 0.625]:
    for _ in range(50):
        pred = pred_base.clone().requires_grad_(True)
        l1 = F.l1_loss(pred, target)
        dsim = d_ssim_ds(ssim_fn, pred, target, scale)
        loss = (1-LAMBDA)*l1 + LAMBDA*dsim
        loss.backward()
    torch.cuda.synchronize()
    times = []
    for _ in range(300):
        pred = pred_base.clone().requires_grad_(True)
        l1 = F.l1_loss(pred, target)
        dsim = d_ssim_ds(ssim_fn, pred, target, scale)
        loss = (1-LAMBDA)*l1 + LAMBDA*dsim
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        loss.backward()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    arr = np.array(times)
    key = f"{scale:.2f}"
    results[key] = {
        "scale": scale,
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "std_ms": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "n_warmup": 50,
        "n_measure": 300,
    }
    print(f"scale={key}: mean={arr.mean():.2f}ms median={np.median(arr):.2f}ms")

with open("results/c42_adaptive/b9/loss_cost_supplementary.json", "w") as f:
    json.dump({"experiment": "B9-A supplementary: scales 0.50 and 0.625", "results": results}, f, indent=2)
print("Saved supplementary results")

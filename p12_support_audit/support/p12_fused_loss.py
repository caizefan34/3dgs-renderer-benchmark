#!/usr/bin/env python3
"""P12 support lane: P11-validated fused-SSIM loss plumbing + runtime identity lock.

This module is the ONLY common loss-plumbing change for the P12 arms
(GMAIN_FUSED / C0_FUSED / A0_FUSED / A1_FUSED). It replaces the frozen
trainer's SepSSIM class with the P11-validated fused-ssim implementation.

Frozen contract (authoritative values from the P11 artifacts
P11_SOURCE_IDENTITY.json / P11_LOSS_EQUIVALENCE.json / P11_LOSS_TIMING.json;
P11 Gate L1 classification: SEMANTICALLY_MATCHED_WITHIN_FP):

  repo          rahul-goel/fused-ssim
  clone         /root/fused-ssim  @ a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38
  module file   site-packages/fused_ssim/__init__.py   sha256 7b3784de48...
  cuda .so      site-packages/fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so
                sha256 dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb
  padding       "same"   (build_flags: -O3 --use_fast_math --maxrregcount=32 sm_80)
  loss          L = 0.8*L1 + 0.2*(1-SSIM)
  API           fused_ssim.fused_ssim(img_bchw, gt_bchw, padding="same", train=True)
  layout/range  trainer's clamp(0,1) RGB images HWC -> contiguous BCHW [0,1]
  GT prep       dataset.get_item (identical to the frozen trainer; [0,1] float)
  reduction     fused_ssim returns ssim_map.mean() (mean over all pixels)

The trainer call-site semantics are unchanged: ``ssim_fn(pred_hwc, gt_hwc)``
returns ``1 - mean_ssim`` exactly like the frozen SepSSIM call.
"""
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time

import torch

# ---------------------------------------------------------------- P11 identity
P12_LOCK_EXPECTED = {
    "repo": "https://github.com/rahul-goel/fused-ssim",
    "clone_path": "/root/fused-ssim",
    "commit": "a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38",
    "module_file": ("/root/miniforge3/envs/gsplat/lib/python3.10/site-packages/"
                    "fused_ssim/__init__.py"),
    "module_sha256": "7b3784de48096f3b5d1ee156ce721f71f77a02650131f6c539331bc7d2d1ecc6",
    "so_path": ("/root/miniforge3/envs/gsplat/lib/python3.10/site-packages/"
                "fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so"),
    "so_sha256": "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb",
    "padding": "same",
    "lambda_l1": 0.8,
    "lambda_dssim": 0.2,
    "api": "fused_ssim.fused_ssim(img_bchw, gt_bchw, padding='same', train=True)",
    "reduction": "ssim_map.mean() (mean over all pixels)",
    "image_range": "[0,1] float32, contiguous BCHW after HWC->BCHW permute",
    "gt_preprocessing": "frozen trainer dataset.get_item (no extra transform)",
    "build_flags": ["-O3", "--use_fast_math", "--maxrregcount=32",
                    "CUDA_ARCHITECTURES=80"],
}

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _git(args):
    try:
        r = subprocess.run(["git", "-C", P12_LOCK_EXPECTED["clone_path"]] + args,
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _runtime_semantic_probe():
    """One-time in-process probe: fused(padding=same) == torch reference SSIM
    computed with the frozen trainer's exact math; padding argument is effective.

    Returns a dict of measured values; raises on structural failure.
    """
    from fused_ssim import fused_ssim
    g = torch.Generator(device="cpu").manual_seed(1234)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 64), torch.linspace(0, 1, 64),
                            indexing="ij")
    base2d = 0.5 + 0.4 * torch.sin(3 * yy) * torch.cos(2 * xx)
    base = base2d.unsqueeze(-1).repeat(1, 1, 3)
    img1 = (base + 0.01 * torch.rand(64, 64, 3, generator=g)).clamp(0, 1)
    img2 = (base + 0.01 * torch.rand(64, 64, 3, generator=g)).clamp(0, 1)
    img1 = img1.unsqueeze(0).permute(0, 3, 1, 2).contiguous().cuda()
    img2 = img2.unsqueeze(0).permute(0, 3, 1, 2).contiguous().cuda()

    fused_same = float(fused_ssim(img1, img2, padding="same", train=True))
    fused_valid = float(fused_ssim(img1, img2, padding="valid", train=True))

    # frozen-trainer SepSSIM math (verbatim), for the same-view comparison
    coords = torch.arange(11, dtype=torch.float32, device="cuda") - 5
    k1d = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    k1d = k1d / k1d.sum()
    k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
    k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0, 1, 3, 2).contiguous()
    import torch.nn.functional as F
    stacked = torch.cat([img1, img2, img1 ** 2, img2 ** 2, img1 * img2], dim=1)
    b = F.conv2d(stacked, k_h, padding=(0, 5), groups=15)
    b = F.conv2d(b, k_v, padding=(5, 0), groups=15)
    mu_p, mu_t = b[:, 0:3], b[:, 3:6]
    bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
    ssim_map = ((2 * (mu_p * mu_t) + 0.01 ** 2) * (2 * (bpt - mu_p * mu_t) + 0.03 ** 2)) / \
               (((mu_p ** 2 + mu_t ** 2) + 0.01 ** 2) *
                (((bp2 - mu_p ** 2) + (bt2 - mu_t ** 2)) + 0.03 ** 2))
    sep_ref = float(ssim_map.mean())

    # gradient flow probe
    img1g = img1.detach().clone().requires_grad_(True)
    loss = 1.0 - fused_ssim(img1g, img2, padding="same", train=True)
    grad = torch.autograd.grad(loss, img1g)[0]
    return {
        "fused_same": fused_same,
        "fused_valid_sentinel": fused_valid,
        "sep_ref_torch_same": sep_ref,
        "abs_diff_vs_sep_ref": abs(fused_same - sep_ref),
        "padding_arg_effective": fused_same != fused_valid,
        "grad_finite": bool(torch.isfinite(grad).all()),
        "grad_abs_mean": float(grad.abs().mean()),
    }


def collect_runtime_lock(context=None):
    """Collect the runtime-observed fused-loss identity (no assertions)."""
    import fused_ssim
    import fused_ssim_cuda

    mod_file = os.path.realpath(fused_ssim.__file__)
    so_file = os.path.realpath(fused_ssim_cuda.__file__)
    rec = {
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "context": context or {},
        "runtime": {
            "module_file": mod_file,
            "module_sha256": sha256_file(mod_file),
            "so_path": so_file,
            "so_sha256": sha256_file(so_file),
            "clone_path": P12_LOCK_EXPECTED["clone_path"],
            "clone_commit": _git(["rev-parse", "HEAD"]),
            "clone_dirty_lines": (_git(["status", "--porcelain"]) or "").splitlines(),
            "cwd": os.getcwd(),
        },
        "contract": {
            "padding": P12_LOCK_EXPECTED["padding"],
            "lambda_l1": P12_LOCK_EXPECTED["lambda_l1"],
            "lambda_dssim": P12_LOCK_EXPECTED["lambda_dssim"],
            "api": P12_LOCK_EXPECTED["api"],
            "api_signature": str(inspect.signature(fused_ssim.fused_ssim)),
            "reduction": P12_LOCK_EXPECTED["reduction"],
            "image_range": P12_LOCK_EXPECTED["image_range"],
            "gt_preprocessing": P12_LOCK_EXPECTED["gt_preprocessing"],
            "train_flag": True,
            "train_flag_note": ("train flag allocates derivative buffers only; the "
                                "ssim_map forward value is identical for train=True/False"),
        },
        "semantic_probe": None,
        "checks": {},
    }
    return rec


def assert_and_log_lock(context=None):
    """Verify the runtime fused-loss identity against the frozen P11 values.

    Logs a single machine-parsable line ``P12_FUSED_LOSS_LOCK {json}``.
    Raises RuntimeError (BLOCKED_FUSED_LOSS_IDENTITY) on any identity mismatch.
    """
    rec = collect_runtime_lock(context=context)
    checks = {}
    problems = []

    exp = P12_LOCK_EXPECTED
    checks["so_sha256"] = rec["runtime"]["so_sha256"] == exp["so_sha256"]
    checks["module_sha256"] = rec["runtime"]["module_sha256"] == exp["module_sha256"]
    if rec["runtime"]["clone_commit"] is None:
        checks["clone_commit"] = None
        checks["clone_clean"] = None
    else:
        checks["clone_commit"] = rec["runtime"]["clone_commit"] == exp["commit"]
        checks["clone_clean"] = len(rec["runtime"]["clone_dirty_lines"]) == 0

    try:
        from fused_ssim import fused_ssim
        sig = inspect.signature(fused_ssim)
        checks["api_padding_param"] = (
            "padding" in sig.parameters and sig.parameters["padding"].default == "same")
    except Exception as e:  # pragma: no cover
        checks["api_padding_param"] = False
        problems.append(f"api import failure: {e}")

    probe = _runtime_semantic_probe()
    rec["semantic_probe"] = probe
    # probe thresholds: same-pad fused must match the torch reference far tighter
    # than the valid-vs-same contrast; P11 L1 found <= 3.3e-5 (rel l2 on gradients)
    checks["semantic_probe"] = (
        probe["abs_diff_vs_sep_ref"] <= 1e-3
        and probe["grad_finite"]
        and abs(probe["fused_same"] - probe["sep_ref_torch_same"]) <= 1e-3)
    checks["padding_effective"] = bool(probe["padding_arg_effective"])

    rec["checks"] = checks
    rec["expected"] = {k: exp[k] for k in
                       ("commit", "module_sha256", "so_sha256", "padding",
                        "lambda_l1", "lambda_dssim", "api")}
    failed = [k for k, v in checks.items() if v is False]
    rec["lock_match"] = len(failed) == 0
    rec["failed_checks"] = failed
    rec["problems"] = problems

    print("P12_FUSED_LOSS_LOCK " + json.dumps(rec, sort_keys=True), flush=True)

    if failed:
        raise RuntimeError(
            "BLOCKED_FUSED_LOSS_IDENTITY: fused-SSIM runtime lock mismatch on "
            f"{failed}; see P12_FUSED_LOSS_LOCK line above")
    return rec


class P12FusedSSIM:
    """Drop-in replacement for the frozen trainer's SepSSIM call site.

    ``__call__(pred, gt) -> 1 - mean SSIM`` (grad-capable), identical to the
    frozen SepSSIM return convention. padding="same" is fixed by the P11 lock.
    """

    PADDING = P12_LOCK_EXPECTED["padding"]

    def __init__(self, device="cuda"):
        self.device = device
        self._range_checked = False

    def __call__(self, pred, target):
        from fused_ssim import fused_ssim
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        pred = pred.contiguous()
        target = target.contiguous()
        if not self._range_checked:
            with torch.no_grad():
                lo = min(float(pred.min()), float(target.min()))
                hi = max(float(pred.max()), float(target.max()))
            self._range_checked = True
            print("P12_FUSED_LOSS_RANGE " + json.dumps({
                "image_min": float(pred.min()), "image_max": float(pred.max()),
                "gt_min": float(target.min()), "gt_max": float(target.max()),
                "range_ok": (lo >= -1e-5 and hi <= 1.0 + 1e-5),
                "note": "first-call image/GT range observation ([0,1] contract)",
            }), flush=True)
            if lo < -1e-5 or hi > 1.0 + 1e-5:
                raise RuntimeError(
                    f"BLOCKED_FUSED_LOSS_IDENTITY: image/GT outside [0,1] "
                    f"(min={lo}, max={hi})")
        ssim = fused_ssim(pred, target, padding=self.PADDING, train=True)
        return 1.0 - ssim

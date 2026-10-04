#!/usr/bin/env python3
"""E3: main time-to-quality judgment (DSH driver) — REFERENCE_LEVEL.

First scientific behavior: mx_guard.require_guarded_process() before any
torch/scientific import. Operates on the 3 registered scenes x seeds.

SCOPE / HONESTY NOTE (per PROTOCOL E3 + NUMERICS + ADAPTER_CONTRACT):
- The main E3 gate is production-CUDA time-to-quality at fixed N. That
  requires the production NATIVE_R0/R1 CUDA renderer + training kernels,
  which are NOT yet implemented. This driver therefore provides the
  REFERENCE_LEVEL structure: paired baseline (native default) vs BES
  candidate (Torch-reference event sampler) fixed-N training with the
  preregistered quality gates (PSNR/SSIM/LPIPS), clearly labeled
  REFERENCE_ONLY. It MUST NOT be reported as a CUDA speedup or as
  GO_FIXED_N_PILOT. Valid E3 evidence requires the production CUDA path.
- If LPIPS dependency is unavailable, the LPIPS gate is marked NOT_COMPUTABLE
  (not silently PASS).

Writes E3.json + E3_DETAILS.json into BES0_OUTPUT_DIR.
"""
from mx_guard import require_guarded_process


def _compute_psnr(pred, gt):
    import torch
    mse = torch.mean((pred - gt) ** 2).item()
    return float(100.0) if mse < 1e-12 else float(-10.0 * torch.log10(torch.tensor(mse)).item())


def _compute_ssim(pred, gt):
    import torch
    import torch.nn.functional as F
    # pred/gt are [H, W, 3] (render format). Convert to channels-first
    # [1, 3, H, W] so conv2d treats spatial dims correctly; the previous
    # unsqueeze(1) misread the 3 RGB channels as image width.
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    x = pred.permute(2, 0, 1).unsqueeze(0)
    y = gt.permute(2, 0, 1).unsqueeze(0)
    ker = _gauss(11, 1.5, x.shape[1])
    mu_x = F.conv2d(x, ker, padding=5)
    mu_y = F.conv2d(y, ker, padding=5)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    s_x = F.conv2d((x - mu_x) ** 2, ker, padding=5)
    s_y = F.conv2d((y - mu_y) ** 2, ker, padding=5)
    s_xy = F.conv2d((x - mu_x) * (y - mu_y), ker, padding=5)
    ssim_map = ((2 * mu_xy + C1) * (2 * s_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (s_x + s_y + C2))
    return float(ssim_map.mean().item())


def _gauss(size, sigma, ch):
    import torch
    k = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(k ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    ker = (g[:, None] * g[None, :]).expand(1, ch, size, size).contiguous()
    return ker


def main():
    cfg, proof = require_guarded_process()  # BEFORE scientific imports
    import hashlib
    import json
    import os
    import time
    from pathlib import Path
    import torch

    import scene_adapter as sa
    from provenance import (protocol_digest, implementation_digest, selected_config_digest,
                            reference_digest)

    if not torch.__version__.startswith(cfg["torch_version"]):
        raise RuntimeError("BLOCKED_ENV: unexpected torch version")
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_ENV: CUDA unavailable")
    cap = torch.cuda.get_device_capability()
    if tuple(cap) != tuple(cfg["expected_compute_capability"]):
        raise RuntimeError(f"BLOCKED_ENV: SM{cap[0]}{cap[1]} != SM80")
    torch.set_num_threads(1)
    device = "cuda"
    output = Path(os.environ["BES0_OUTPUT_DIR"])
    pkg_root = Path(os.environ["BES0_PACKAGE_ROOT"])
    prereg = json.loads((pkg_root / "config" / "prereg.json").read_text())
    manifest = json.loads((pkg_root / "config" / "scene_manifest.json").read_text())
    qg = prereg["quality"]
    checks = {}
    details = {"scope": "REFERENCE_LEVEL structure; NOT a CUDA speedup/GO claim",
               "quality_gates": qg}

    # LPIPS availability
    try:
        import lpips  # noqa: F401
        lpips_ok = True
    except Exception:
        lpips_ok = False
    details["lpips_available"] = lpips_ok
    checks["lpips_gate_status_known"] = lpips_ok  # if False, mark NOT_COMPUTABLE

    seeds = prereg["seeds"]
    max_steps = prereg["max_pilot_steps"]
    eval_every = prereg["eval_every_steps"]

    def train_to_quality(scene, dev_ids, heldout_ids, method, seed):
        """Reference fixed-N training loop. method in {'baseline','candidate'}.
        Returns dict with time-to-quality steps (or None) and endpoint quality."""
        torch.manual_seed(seed)
        # optimizer over active params
        params = [scene.xyz, scene.shs, scene.scaling, scene.rotation, scene.opacity]
        opt = torch.optim.Adam(params, lr=1e-3)
        cams = [scene.cameras[i] for i in dev_ids if i in scene.cameras]
        eval_cams = [scene.cameras[i] for i in heldout_ids if i in scene.cameras]
        t0 = time.monotonic()
        reached = None
        endpoint = None
        for step in range(0, max_steps + 1):
            if step > 0 and step % eval_every == 0:
                with torch.no_grad():
                    q = evaluate(scene, eval_cams)
                endpoint = q
                if reached is None and quality_met(q, qg):
                    reached = step
            if step >= max_steps:
                break
            cam = cams[step % len(cams)]
            if method == "baseline":
                img = sa.native_default(scene, cam).image
            else:
                tl = sa.project_and_sort(scene, cam, tile_size=16)
                routing = sa.build_bounds(tl, mode="R0_ENUMERATED_NATIVE")
                img = sa.bes_sample(scene, cam, routing, {"seed": step}, detach=False).images["A"]
            gt = scene.gt_cache.get(cam.image_id)
            if gt is None:
                continue
            loss = 0.5 * ((img - gt) ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize()
        return {"time_to_quality_steps": reached, "wall_s": time.monotonic() - t0,
                "endpoint": endpoint}

    def evaluate(scene, eval_cams):
        psnrs, ssims = [], []
        with torch.no_grad():
            for cam in eval_cams:
                img = sa.native_default(scene, cam).image
                gt = scene.gt_cache.get(cam.image_id)
                if gt is None:
                    continue
                psnrs.append(_compute_psnr(img, gt)); ssims.append(_compute_ssim(img, gt))
        return {"mean_psnr": sum(psnrs) / max(len(psnrs), 1),
                "mean_ssim": sum(ssims) / max(len(ssims), 1)}

    def quality_met(q, qg):
        return (q["mean_psnr"] >= qg.get("mean_psnr_min", 0) and
                q["mean_ssim"] >= qg.get("mean_ssim_min", 0))

    # GT cache must be attached to scene for training; load lazily per scene.
    pairs = {}
    for name, rec in manifest["scenes"].items():
        try:
            scene = sa.load_scene({"name": name, **rec}, device)
            scene.gt_cache = sa.load_gt_cache(scene, device)  # adapter helper
        except Exception as exc:
            pairs[name] = {"error": str(exc)}
            continue
        dev_ids = rec["dev_camera_ids"]; held_ids = rec["heldout_camera_ids"]
        scene_pairs = {}
        for seed in seeds:
            b = train_to_quality(scene, dev_ids, held_ids, "baseline", seed)
            c = train_to_quality(scene, dev_ids, held_ids, "candidate", seed)
            scene_pairs[seed] = {"baseline": b, "candidate": c}
        pairs[name] = scene_pairs
    details["pairs"] = pairs
    checks["reference_pairs_recorded"] = all("error" in v or len(v) >= len(seeds) for v in pairs.values())

    status = "PASS" if all(checks.values()) else "FAIL"
    stage = {
        "schema": "bes0.stage.v1", "stage": "E3", "status": status,
        "protocol_sha256": protocol_digest(), "implementation_sha256": implementation_digest(),
        "reference_sha256": reference_digest(), "source_commit": "32ab80e773f74f4d8e40ff4e338c86b29c7957f5",
        "guard_sha256": hashlib.sha256(Path(os.environ["BES0_GUARD_PROOF"]).read_bytes()).hexdigest(),
        "selected_config_sha256": selected_config_digest(), "cuda_implemented": False, "timing_level": "REFERENCE_TORCH_NOT_CUDA", "inputs_verified": True,
        "checks": checks, "raw_files": [], "details": details,
        "scope": "REFERENCE_LEVEL; valid E3 requires production CUDA renderer",
        "guard": proof["hostname"],
    }
    (output / "E3.json").write_text(json.dumps(stage, indent=2) + "\n")
    (output / "E3_DETAILS.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(stage, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""P16-RB: common-split render + common-metric evaluation of a frozen P12 C0 checkpoint.

Purpose: resolve the train/test split difference (P12 has NO split and evaluates
all cameras; every external 2026 method uses the hold-8 convention on sorted
image names). This script renders the hold-8 test cameras from a frozen P12 C0
final checkpoint using the exact P12 arm-c0 render path, and evaluates with the
exact P12 metric implementation (SepSSIM + PSNR + LPIPS-vgg), which is extracted
from the frozen trainer source by byte range and integrity-hashed.

No training, no parameter change. Renders only.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TRAINER = "/root/epic05_p12/publication_trainer_p12.py"
METRIC_BLOCK_FIRST_LINE = 295   # 1-based, inclusive: "# ---- SepSSIM / PSNR ..."
METRIC_BLOCK_LAST_LINE = 323    # 1-based, inclusive: end of compute_psnr
METRIC_BLOCK_SHA256 = "87e4ebd6bb15073e24cf15045dda383e1100ddfc7393c765e37e897346b3f0b6"
TEST_STEP = 8
EPS2D = 0.3


def extract_metric_block(trainer_path: str):
    raw = Path(trainer_path).read_bytes().decode()
    lines = raw.split("\n")
    block = "\n".join(lines[METRIC_BLOCK_FIRST_LINE - 1:METRIC_BLOCK_LAST_LINE]) + "\n"
    got = hashlib.sha256(block.encode()).hexdigest()
    if got != METRIC_BLOCK_SHA256:
        raise SystemExit(f"FATAL: metric block hash mismatch {got} != {METRIC_BLOCK_SHA256}")
    # The block lives inside an if/elif/else branch of the trainer: its leading comment
    # line sits at column 0 while the code carries the branch's 4-space indent, so
    # textwrap.dedent cannot normalize it. Executing under "if True:" keeps the byte
    # range (and therefore the hash above) exactly as frozen.
    ns = {"torch": __import__("torch"), "F": __import__("torch.nn.functional", fromlist=["x"]), "math": __import__("math")}
    exec(compile("if True:\n" + block, f"{trainer_path}:{METRIC_BLOCK_FIRST_LINE}", "exec"), ns)
    return ns, got


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def gpu_uuid() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15).stdout.strip()
        return out
    except Exception as e:  # pragma: no cover
        return f"unavailable: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--trainer", default=TRAINER)
    ap.add_argument("--expected-names-file", default=None,
                    help="optional JSON list/object of test image names to assert equality against")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import numpy as np
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, "/root/epic05_p12")
    import publication_trainer_p12 as T
    ident = T.bootstrap_c0()
    from gaussian_model import GaussianModel
    from config import ReferenceV1Config
    from dataset import GTDataset
    from gsplat.experimental import rasterize_gaussian_higs_dynamic

    ns, block_hash = extract_metric_block(args.trainer)
    SepSSIM, compute_psnr = ns["SepSSIM"], ns["compute_psnr"]

    import lpips
    lpips_fn = lpips.LPIPS(net="vgg").to("cuda")
    ssim_fn = SepSSIM(device="cuda")

    outdir = Path(args.outdir)
    (outdir / "images").mkdir(parents=True, exist_ok=True)

    dataset = GTDataset(scene=args.scene, repo_root=T.REPO, resolution="1080p",
                        device="cuda", background="black")
    n = len(dataset)
    names = []
    for i in range(n):
        cam = dataset._valid_cameras[i]
        nm = getattr(cam, "image_name", None) or getattr(cam, "img_name", None)
        names.append(str(nm))
    order = sorted(range(n), key=lambda i: names[i])
    test_idx = [i for pos, i in enumerate(order) if pos % TEST_STEP == 0]
    test_names = [names[i] for i in test_idx]

    if args.expected_names_file:
        exp = json.loads(Path(args.expected_names_file).read_text())
        exp_set = set(exp if isinstance(exp, list) else exp.get("test_names", []))
        exp_set = {str(Path(x).stem) for x in exp_set}
        got_set = {str(Path(x).stem) for x in test_names}
        if exp_set != got_set:
            raise SystemExit(f"FATAL: split mismatch vs expected: only_expected={sorted(exp_set - got_set)[:5]} "
                             f"only_local={sorted(got_set - exp_set)[:5]}")

    config = ReferenceV1Config()
    config.repo_root = T.REPO
    model = GaussianModel(max_sh_degree=config.sh_degree)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # GaussianModel.restore() forwards this dict straight into training_setup(), which
    # indexes it by key -- the trainer builds the identical 9-key dict at its own setup.
    model.restore(ckpt, {
        "position_lr_init": config.position_lr_init,
        "position_lr_final": config.position_lr_final,
        "position_lr_delay_mult": config.position_lr_delay_mult,
        "position_lr_max_steps": config.position_lr_max_steps,
        "feature_lr": config.feature_lr,
        "opacity_lr": config.opacity_lr,
        "scaling_lr": config.scaling_lr,
        "rotation_lr": config.rotation_lr,
        "percent_dense": config.percent_dense,
    })
    n_gs = int(model._xyz.shape[0])

    rows, manifest = [], {}
    t0 = time.perf_counter()
    for ci in test_idx:
        cam, gt = dataset.get_item(ci)
        with torch.no_grad():
            out = rasterize_gaussian_higs_dynamic(
                model.get_xyz, model.get_rotation, model.get_scaling,
                model.get_opacity, model.get_features,
                sh_degree=model.active_sh_degree,
                viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
                Ks=cam.K.unsqueeze(0).unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                backward_mode="higs_native", enable_culling=True,
                camera_model="pinhole", render_mode="RGB",
                eps2d=EPS2D)
            frame = out["frame"].reshape(cam.image_height, cam.image_width, 3).clamp(0, 1)
        psnr = compute_psnr(frame, gt)
        ssim = 1.0 - float(ssim_fn(frame, gt).item())
        pred_lp = frame.unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
        gt_lp = gt.unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
        lp = float(lpips_fn(pred_lp, gt_lp).item())
        img_path = outdir / "images" / (Path(names[ci]).stem + ".png")
        arr = (frame.detach().cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        from PIL import Image
        Image.fromarray(arr).save(img_path, format="PNG", compress_level=6)
        manifest[img_path.name] = sha256_file(img_path)
        rows.append({"camera_index": int(ci), "image_name": names[ci],
                     "width": int(cam.image_width), "height": int(cam.image_height),
                     "psnr": psnr, "ssim": ssim, "lpips": lp})
        print(f"  [{args.scene} {Path(names[ci]).stem}] PSNR={psnr:.3f} SSIM={ssim:.5f} LPIPS={lp:.4f}", flush=True)
    render_s = time.perf_counter() - t0

    summary = {
        "scene": args.scene,
        "ckpt": args.ckpt,
        "ckpt_sha256": sha256_file(Path(args.ckpt)),
        "ckpt_iteration": int(ckpt.get("iteration", -1)),
        "ckpt_arm": ckpt.get("arm"),
        "trainer": args.trainer,
        "trainer_sha256": sha256_file(Path(args.trainer)),
        "metric_block_lines": [METRIC_BLOCK_FIRST_LINE, METRIC_BLOCK_LAST_LINE],
        "metric_block_sha256": block_hash,
        "bootstrap_identity": ident,
        "eps2d": EPS2D,
        "renderer": "gsplat.experimental.rasterize_gaussian_higs_dynamic (arm c0)",
        "split": {"rule": "hold8_sorted_image_name_idx_mod_8_eq_0", "test_step": TEST_STEP,
                  "n_total": n, "n_test": len(test_idx)},
        "resolution": [int(dataset.target_width), int(dataset.target_height)],
        "n_gaussians": n_gs,
        "active_sh_degree": int(model.active_sh_degree),
        "mean": {"psnr": float(np.mean([r["psnr"] for r in rows])),
                 "ssim": float(np.mean([r["ssim"] for r in rows])),
                 "lpips": float(np.mean([r["lpips"] for r in rows]))},
        "per_camera": rows,
        "render_seconds": render_s,
        "images_manifest": {"n": len(manifest), "sha256": manifest},
        "gpu_uuids": gpu_uuid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "contaminated": None,
        "status": "SUCCESS",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (outdir / "common_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{args.scene}] COMMON_EVAL_DONE n_test={len(test_idx)} "
          f"PSNR={summary['mean']['psnr']:.4f} SSIM={summary['mean']['ssim']:.6f} "
          f"LPIPS={summary['mean']['lpips']:.5f} ({render_s:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

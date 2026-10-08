#!/usr/bin/env python3
"""P16-RB common evaluation: ONE metric pipeline for every method and for P12_C0.

Metrics are computed with the frozen P12 metric block (SepSSIM + PSNR) plus
LPIPS(vgg), on 8-bit PNG renders against the common 1920x1080 GT images, for exactly
the hold-8 test split. Every rendered and GT image is sha256-hashed into a manifest.

Using one code path on identical-format inputs is what makes metrics_common comparable
across methods (P12_C0 included).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/epic05_p16rb/scripts")
from p16_p12_render_common import extract_metric_block, TRAINER  # noqa: E402

COMMON = Path("/root/epic05_p16rb/common_data/mipnerf360")
ART = Path("/root/epic05_p16rb/artifacts")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def find_render(rgb_dir: Path, stem: str):
    for ext in (".png", ".jpg", ".jpeg", ".JPG", ".PNG"):
        p = rgb_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--rgb-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--expected-names", default=None)
    ap.add_argument("--name-map", default=None,
                    help="JSON from p16_name_map.py: framework render file -> common test stem; "
                         "required when the framework writes index-based render names")
    args = ap.parse_args()

    import numpy as np
    import torch
    import lpips
    from PIL import Image

    gt_dir = Path(args.gt_dir) if args.gt_dir else COMMON / args.scene / "images"
    exp_file = Path(args.expected_names) if args.expected_names else ART / f"expected_names_{args.scene}.json"
    exp = json.loads(exp_file.read_text())
    expected = [str(Path(x).stem) for x in exp["test_names"]]

    ns, block_hash = extract_metric_block(TRAINER)
    SepSSIM, compute_psnr = ns["SepSSIM"], ns["compute_psnr"]
    ssim_fn = SepSSIM(device="cuda")
    lpips_fn = lpips.LPIPS(net="vgg").to("cuda")

    rgb_dir = Path(args.rgb_dir)
    nmap_info = None
    if args.name_map:
        nmap_path = Path(args.name_map)
        nmap_doc = json.loads(nmap_path.read_text())
        nmap = nmap_doc["name_map"]
        if sorted(nmap.values()) != sorted(expected):
            raise SystemExit("FATAL: name map stems do not equal the expected hold-8 names")
        nmap_info = {"path": str(nmap_path), "status": nmap_doc.get("status"),
                     "sha256": sha256_file(nmap_path)}
        items = [(stem, rgb_dir / rf) for rf, stem in sorted(nmap.items(), key=lambda kv: kv[1])]
    else:
        items = [(stem, find_render(rgb_dir, stem)) for stem in expected]

    rows, manifest, missing = [], {"renders": {}, "gt": {}}, []
    for stem, rp in items:
        gp = gt_dir / f"{stem}.png"
        if rp is None or not rp.exists() or not gp.exists():
            missing.append({"stem": stem, "render": str(rp) if rp else None,
                            "gt_exists": gp.exists()})
            continue
        with Image.open(rp) as im:
            arr = np.asarray(im.convert("RGB")).astype(np.float32) / 255.0
        with Image.open(gp) as im:
            gt_arr = np.asarray(im.convert("RGB")).astype(np.float32) / 255.0
        if arr.shape != gt_arr.shape:
            missing.append({"stem": stem, "render": str(rp), "shape_mismatch": [list(arr.shape), list(gt_arr.shape)]})
            continue
        frame = torch.from_numpy(arr).cuda()
        gt = torch.from_numpy(gt_arr).cuda()
        psnr = compute_psnr(frame, gt)
        ssim = 1.0 - float(ssim_fn(frame, gt).item())
        pred_lp = frame.unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
        gt_lp = gt.unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
        lp = float(lpips_fn(pred_lp, gt_lp).item())
        manifest["renders"][rp.name] = sha256_file(rp)
        manifest["gt"][gp.name] = sha256_file(gp)
        rows.append({"stem": stem, "render_file": rp.name, "width": int(arr.shape[1]),
                     "height": int(arr.shape[0]), "psnr": psnr, "ssim": ssim, "lpips": lp})

    res_seen = sorted({(r["width"], r["height"]) for r in rows})
    out = {
        "method": args.method, "scene": args.scene,
        "gt_dir": str(gt_dir), "rgb_dir": str(rgb_dir),
        "expected_names_file": str(exp_file),
        "metric_source": {"trainer": TRAINER, "metric_block_sha256": block_hash,
                          "lpips": "lpips.LPIPS(net='vgg')", "input": "8-bit PNG /255 float"},
        "name_map": nmap_info,
        "n_expected": len(expected), "n_evaluated": len(rows), "n_missing": len(missing),
        "missing": missing,
        "resolution_seen": [list(x) for x in res_seen],
        "mean": ({"psnr": float(np.mean([r["psnr"] for r in rows])),
                  "ssim": float(np.mean([r["ssim"] for r in rows])),
                  "lpips": float(np.mean([r["lpips"] for r in rows]))} if rows else None),
        "per_image": rows,
        "manifest": manifest,
        "status": "SUCCESS" if rows and not missing else "INCOMPLETE",
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"method": args.method, "scene": args.scene, "n_evaluated": len(rows),
                      "n_missing": len(missing), "resolution_seen": out["resolution_seen"],
                      "mean": out["mean"], "status": out["status"]}, indent=2))
    return 0 if out["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())

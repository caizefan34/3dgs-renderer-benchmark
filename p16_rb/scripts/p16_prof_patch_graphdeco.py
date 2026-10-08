#!/usr/bin/env python3
"""P16-RB: apply timing-only instrumentation to a COPIED FastGS or SADGS tree.

Span enter/exit statements are placed so that both always execute together
(enters at loop-body level, exits at the same level right after the region),
which keeps the profiling context balanced on every iteration.
Every edit is an exact-anchor replacement; uniqueness is enforced. Writes patch_report.json.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


COMMON = [
    ("total_time = 0.0\n",
     "total_time = 0.0\n    p16_prof.P16.init()\n",
     "init before loop"),
    ("    for iteration in range(first_iter, opt.iterations + 1):\n",
     "    for iteration in range(first_iter, opt.iterations + 1):\n        p16_prof.P16.iter_begin(iteration)\n",
     "iteration marker at loop head"),
    ("    # scene.save(iteration)\n",
     "    p16_prof.P16.close()\n    # scene.save(iteration)\n",
     "close after training loop"),
    ("            # record time\n            optim_end.record()\n",
     "            _p16o.__exit__(None, None, None)\n            # record time\n            optim_end.record()\n",
     "optimizer span exit (always reached)"),
]

FASTGS = [
    ("        # Pick a random Camera\n",
     "        _p16a = p16_prof.P16.span('data'); _p16a.__enter__()\n        # Pick a random Camera\n",
     "data span enter"),
    ("        _ = viewpoint_indices.pop(rand_idx)\n",
     "        _ = viewpoint_indices.pop(rand_idx)\n        _p16a.__exit__(None, None, None)\n",
     "data span exit"),
    ("        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)\n",
     "        _p16f = p16_prof.P16.span('raster_forward'); _p16f.__enter__()\n"
     "        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)\n"
     "        _p16f.__exit__(None, None, None)\n",
     "raster_forward span"),
    ("        # Loss\n",
     "        _p16l = p16_prof.P16.span('loss'); _p16l.__enter__()\n        # Loss\n",
     "loss span enter"),
    ("        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)\n",
     "        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)\n"
     "        _p16l.__exit__(None, None, None)\n",
     "loss span exit"),
    ("        loss.backward()\n",
     "        _p16b = p16_prof.P16.span('backward'); _p16b.__enter__()\n"
     "        loss.backward()\n"
     "        _p16b.__exit__(None, None, None)\n",
     "backward span"),
    ("            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log\n",
     "            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log\n"
     "            p16_prof.P16.log_iter(iteration, viewpoint_cam.image_name, loss.item())\n",
     "loss/cam log"),
    ("            # Densification\n",
     "            _p16d = p16_prof.P16.span('density'); _p16d.__enter__()\n            # Densification\n",
     "density span enter"),
    ("            # The multiview consistent pruning of fastgs. We do it every 3k iterations after 15k\n",
     "            _p16d.__exit__(None, None, None)\n"
     "            # The multiview consistent pruning of fastgs. We do it every 3k iterations after 15k\n",
     "density span exit (before final-prune block)"),
    ("            if iteration % 3000 == 0 and iteration > 15_000 and iteration < 30_000:\n                my_viewpoint_stack = scene.getTrainCameras().copy()\n",
     "            if iteration % 3000 == 0 and iteration > 15_000 and iteration < 30_000:\n"
     "                _p16p = p16_prof.P16.span('density'); _p16p.__enter__()\n"
     "                my_viewpoint_stack = scene.getTrainCameras().copy()\n",
     "final-prune span enter (inside guard)"),
    ("                gaussians.final_prune_fastgs(min_opacity = 0.1, pruning_score = pruning_score)\n",
     "                gaussians.final_prune_fastgs(min_opacity = 0.1, pruning_score = pruning_score)\n"
     "                _p16p.__exit__(None, None, None)\n",
     "final-prune span exit (inside guard)"),
    ("            # Optimization step\n",
     "            _p16o = p16_prof.P16.span('optimizer'); _p16o.__enter__()\n            # Optimization step\n",
     "optimizer span enter"),
]

SADGS = [
    ("            # Pop from the start of the FPS-sorted list\n            if not viewpoint_stack:\n",
     "            _p16a = p16_prof.P16.span('data'); _p16a.__enter__()\n"
     "            # Pop from the start of the FPS-sorted list\n            if not viewpoint_stack:\n",
     "data span enter (batch loop)"),
    ("            viewpoint_cam = viewpoint_stack.pop(0)\n",
     "            viewpoint_cam = viewpoint_stack.pop(0)\n"
     "            _p16a.__exit__(None, None, None)\n"
     "            _p16cams.append(viewpoint_cam.image_name)\n",
     "data span exit + cam log"),
    ("            render_pkg = render_structgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)\n",
     "            _p16f = p16_prof.P16.span('raster_forward'); _p16f.__enter__()\n"
     "            render_pkg = render_structgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)\n"
     "            _p16f.__exit__(None, None, None)\n",
     "raster_forward span"),
    ("            gt_image = viewpoint_cam.original_image.cuda()\n",
     "            _p16l = p16_prof.P16.span('loss'); _p16l.__enter__()\n"
     "            gt_image = viewpoint_cam.original_image.cuda()\n",
     "loss span enter"),
    ("            rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)+ opt.lambda_l2 * Ll2\n",
     "            rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)+ opt.lambda_l2 * Ll2\n"
     "            _p16l.__exit__(None, None, None)\n",
     "loss span exit"),
    ("            loss.backward()\n",
     "            _p16b = p16_prof.P16.span('backward'); _p16b.__enter__()\n"
     "            loss.backward()\n"
     "            _p16b.__exit__(None, None, None)\n",
     "backward span"),
    ("        last_visibility_filter = None\n",
     "        last_visibility_filter = None\n        _p16cams = []\n",
     "per-iteration cam list init"),
    ("                if iteration < opt.densify_until_iter and iteration % 10 == 0:\n",
     "                if iteration < opt.densify_until_iter and iteration % 10 == 0:\n"
     "                    _p16s = p16_prof.P16.span('density_stats'); _p16s.__enter__()\n",
     "freq-stats span enter (inside guard)"),
    ("                    update_freq_stats_online(viewpoint_cam, gaussians, cov2D, visibility_filter, structure_tensor_cache, viewspace_point_tensor=viewspace_point_tensor, grad_threshold= opt.freq_grad_threshold, transmittance_threshold=opt.freq_transmittance_threshold, opacity_threshold=opt.freq_opacity_threshold, eta_compute_mode=opt.eta_compute_mode)\n",
     "                    update_freq_stats_online(viewpoint_cam, gaussians, cov2D, visibility_filter, structure_tensor_cache, viewspace_point_tensor=viewspace_point_tensor, grad_threshold= opt.freq_grad_threshold, transmittance_threshold=opt.freq_transmittance_threshold, opacity_threshold=opt.freq_opacity_threshold, eta_compute_mode=opt.eta_compute_mode)\n"
     "                    _p16s.__exit__(None, None, None)\n",
     "freq-stats span exit (inside guard)"),
    ("            ema_loss_for_log = 0.4 * (batch_loss / opt.batch_size) + 0.6 * ema_loss_for_log\n",
     "            ema_loss_for_log = 0.4 * (batch_loss / opt.batch_size) + 0.6 * ema_loss_for_log\n"
     "            p16_prof.P16.log_iter(iteration, '+'.join(_p16cams), batch_loss / opt.batch_size)\n",
     "loss/cam log"),
    ("            # Densification\n",
     "            _p16d = p16_prof.P16.span('density'); _p16d.__enter__()\n            # Densification\n",
     "density span enter"),
    ("            # Optimization step\n",
     "            _p16d.__exit__(None, None, None)\n"
     "            _p16o = p16_prof.P16.span('optimizer'); _p16o.__enter__()\n"
     "            # Optimization step\n",
     "density span exit + optimizer span enter"),
]


def insert_import(text: str) -> str:
    if "import p16_prof" in text:
        return text
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if re.match(r"^(import |from )", ln):
            lines.insert(i, "import p16_prof")
            return "\n".join(lines)
    raise SystemExit("no import statement found to anchor the injection")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="copied FastGS/SADGS tree root")
    ap.add_argument("--method", required=True, choices=["FastGS", "SADGS"])
    ap.add_argument("--prof-module", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    tree = Path(args.tree)
    train = tree / "train.py"
    if not train.is_file():
        raise SystemExit(f"{train} not found")

    dst = tree / "p16_prof.py"
    dst.write_bytes(Path(args.prof_module).read_bytes())

    edits = COMMON + (FASTGS if args.method == "FastGS" else SADGS)
    text = train.read_text(encoding="utf-8")
    applied = []
    for old, new, desc in edits:
        hits = text.count(old)
        if hits != 1:
            raise SystemExit(f"anchor not unique ({hits}) for [{desc}]: {old[:90]!r}")
        text = text.replace(old, new)
        applied.append(desc)
    before_import = text
    text = insert_import(text)
    train.write_text(text, encoding="utf-8")

    report = {
        "tree": str(tree), "method": args.method,
        "prof_module_sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
        "train_py_sha256_after": hashlib.sha256(train.read_bytes()).hexdigest(),
        "edits_applied": applied,
        "import_injected": text != before_import,
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"patched": str(train), "edits": len(applied), "report": args.report}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

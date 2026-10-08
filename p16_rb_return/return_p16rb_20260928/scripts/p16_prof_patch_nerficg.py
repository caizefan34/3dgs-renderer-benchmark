#!/usr/bin/env python3
"""P16-RB: apply timing-only instrumentation to a COPIED nerficg tree.

Every edit is an exact-anchor line insertion (no reindentation of existing code).
Fails loudly if any anchor is not found exactly once. Writes patch_report.json.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

R = []


def add(fname, old, new, desc):
    R.append({"file": fname, "old": old, "new": new, "desc": desc, "hits": 0})


# ---------------------------------------------------------------- Base trainer
BASE = "src/Methods/Base/Trainer.py"
add(BASE,
    "            training_callbacks = self._gather_callbacks(0)\n",
    "            training_callbacks = self._gather_callbacks(0)\n            p16_prof.P16.init()\n",
    "init before main loop")
add(BASE,
    "            for iteration in Logger.log_progress(range(starting_iteration, self.NUM_ITERATIONS), desc='training', miniters=10):\n",
    "            for iteration in Logger.log_progress(range(starting_iteration, self.NUM_ITERATIONS), desc='training', miniters=10):\n                p16_prof.P16.iter_begin(iteration)\n",
    "iteration marker at loop head")
add(BASE,
    "                    with callback.timer:\n                        callback(self, iteration, dataset)\n",
    "                    with callback.timer:\n                        _p16cb = p16_prof.P16.span('cb_' + callback.__name__); _p16cb.__enter__()\n"
    "                        callback(self, iteration, dataset)\n"
    "                        _p16cb.__exit__(None, None, None)\n",
    "per-callback span")
add(BASE,
    "        Logger.log('training finished successfully')\n",
    "        p16_prof.P16.close()\n        Logger.log('training finished successfully')\n",
    "close after training")

# ---------------------------------------------------------------- FasterGS
FG = "src/Methods/FasterGS/Trainer.py"
add(FG,
    "        # get random view\n        view = self.train_sampler.get(dataset=dataset)['view']\n",
    "        # get random view\n        _p16a = p16_prof.P16.span('data'); _p16a.__enter__()\n"
    "        view = self.train_sampler.get(dataset=dataset)['view']\n"
    "        _p16a.__exit__(None, None, None)\n",
    "data span (FasterGS)")
add(FG,
    "        image = self.renderer.render_image_training(\n"
    "            view=view,\n"
    "            update_densification_info=not self.USE_MCMC and iteration < self.DENSIFICATION_END_ITERATION,\n"
    "            bg_color=bg_color,\n"
    "        )\n",
    "        _p16f = p16_prof.P16.span('raster_forward'); _p16f.__enter__()\n"
    "        image = self.renderer.render_image_training(\n"
    "            view=view,\n"
    "            update_densification_info=not self.USE_MCMC and iteration < self.DENSIFICATION_END_ITERATION,\n"
    "            bg_color=bg_color,\n"
    "        )\n"
    "        _p16f.__exit__(None, None, None)\n",
    "raster_forward span (FasterGS)")
add(FG,
    "            rgb_gt = apply_background_color(rgb_gt, alpha_gt, bg_color)\n        loss = self.loss(image, rgb_gt)\n",
    "            rgb_gt = apply_background_color(rgb_gt, alpha_gt, bg_color)\n"
    "        _p16l = p16_prof.P16.span('loss'); _p16l.__enter__()\n"
    "        loss = self.loss(image, rgb_gt)\n"
    "        _p16l.__exit__(None, None, None)\n"
    "        p16_prof.P16.log_iter(iteration, view.camera_index, loss.detach().float().item())\n",
    "loss span + loss/cam log (FasterGS)")
add(FG,
    "        # backward\n        loss.backward()\n",
    "        # backward\n        _p16b = p16_prof.P16.span('backward'); _p16b.__enter__()\n        loss.backward()\n        _p16b.__exit__(None, None, None)\n",
    "backward span (FasterGS)")
add(FG,
    "        # optimizer step\n"
    "        self.model.gaussians.optimizer.step()\n"
    "        self.model.gaussians.optimizer.zero_grad()\n"
    "        self.model.gaussians.post_optimizer_step(inject_noise=self.USE_MCMC)\n",
    "        _p16o = p16_prof.P16.span('optimizer'); _p16o.__enter__()\n"
    "        # optimizer step\n"
    "        self.model.gaussians.optimizer.step()\n"
    "        self.model.gaussians.optimizer.zero_grad()\n"
    "        self.model.gaussians.post_optimizer_step(inject_noise=self.USE_MCMC)\n"
    "        _p16o.__exit__(None, None, None)\n",
    "optimizer span (FasterGS)")

# ---------------------------------------------------------------- FasterGSFused
FU = "src/Methods/FasterGSFused/Trainer.py"
add(FU,
    "        # get random view\n        view = self.train_sampler.get(dataset=dataset)['view']\n",
    "        # get random view\n        _p16a = p16_prof.P16.span('data'); _p16a.__enter__()\n"
    "        view = self.train_sampler.get(dataset=dataset)['view']\n"
    "        _p16a.__exit__(None, None, None)\n",
    "data span (Fused)")
add(FU,
    "        image, autograd_dummy = self.renderer.render_image_training(\n"
    "            view=view,\n"
    "            update_densification_info=iteration < self.DENSIFICATION_END_ITERATION,\n"
    "            bg_color=bg_color,\n"
    "            adam_step_count=optimization_step,\n"
    "            autograd_dummy=self.autograd_dummy,\n"
    "        )\n",
    "        _p16f = p16_prof.P16.span('raster_forward'); _p16f.__enter__()\n"
    "        image, autograd_dummy = self.renderer.render_image_training(\n"
    "            view=view,\n"
    "            update_densification_info=iteration < self.DENSIFICATION_END_ITERATION,\n"
    "            bg_color=bg_color,\n"
    "            adam_step_count=optimization_step,\n"
    "            autograd_dummy=self.autograd_dummy,\n"
    "        )\n"
    "        _p16f.__exit__(None, None, None)\n",
    "raster_forward span (Fused)")
add(FU,
    "        loss = self.loss(image, rgb_gt) + 0.0 * autograd_dummy\n",
    "        _p16l = p16_prof.P16.span('loss'); _p16l.__enter__()\n"
    "        loss = self.loss(image, rgb_gt) + 0.0 * autograd_dummy\n"
    "        _p16l.__exit__(None, None, None)\n"
    "        p16_prof.P16.log_iter(iteration, view.camera_index, loss.detach().float().item())\n",
    "loss span + loss/cam log (Fused)")
add(FU,
    "        loss.backward()\n",
    "        _p16b = p16_prof.P16.span('backward'); _p16b.__enter__()\n"
    "        loss.backward()\n"
    "        _p16b.__exit__(None, None, None)\n",
    "backward span (Fused)")


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
    ap.add_argument("--tree", required=True, help="copied nerficg tree root")
    ap.add_argument("--prof-module", required=True, help="path to p16_prof.py")
    ap.add_argument("--report", required=True, help="patch_report.json output path")
    args = ap.parse_args()

    tree = Path(args.tree)
    src = tree / "src"
    if not src.is_dir():
        raise SystemExit(f"{src} not a nerficg src dir")

    dst = src / "p16_prof.py"
    dst.write_bytes(Path(args.prof_module).read_bytes())

    report = {"tree": str(tree), "prof_module_sha256": hashlib.sha256(dst.read_bytes()).hexdigest(), "files": []}
    touched = {}
    for r in R:
        p = tree / r["file"]
        if not p.exists():
            raise SystemExit(f"missing file: {p}")
        text = p.read_text(encoding="utf-8")
        r["hits"] = text.count(r["old"])
        if r["hits"] != 1:
            raise SystemExit(f"anchor not unique ({r['hits']}) for [{r['desc']}] in {r['file']}: {r['old'][:80]!r}")
        text = text.replace(r["old"], r["new"])
        if p not in touched:
            touched[p] = {"sha256_before": hashlib.sha256(p.read_bytes()).hexdigest(), "edits": []}
        p.write_text(text, encoding="utf-8")
        touched[p]["edits"].append(r["desc"])

    for p, info in touched.items():
        info["file"] = str(p.relative_to(tree))
        info["sha256_after"] = hashlib.sha256(p.read_bytes()).hexdigest()
        report["files"].append(info)

    # import injection for the three patched files
    for rel in (BASE, FG, FU):
        p = tree / rel
        text = insert_import(p.read_text(encoding="utf-8"))
        p.write_text(text, encoding="utf-8")
        report["files"].append({"file": rel, "import_injected": True,
                                "sha256_after": hashlib.sha256(p.read_bytes()).hexdigest()})

    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"patched_files": len(report["files"]), "edits": len(R), "report": args.report}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

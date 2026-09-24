#!/usr/bin/env python3
"""
init_run_dirs.py — Create the empty per-run artifact directory scaffold for the FINAL 30K benchmark.

Creates artifacts/final-30k/<candidate>/<scene>/ for every candidate x scene in the
frozen manifest, with TEMPLATE (empty/schema) versions of the required per-run files.
Does NOT populate any numbers. The benchmark run itself fills these in.

Candidates default to the frozen registry keys; scenes come from manifest.json.
P2_FINAL_V2 dirs are created too (so the harness accepts it later without protocol change),
marked status=PENDING in their template final_status.json.
"""
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL30K = REPO_ROOT / "artifacts" / "final-30k"

CANDIDATES = ["B1A_ACCUTILE", "C0_V3", "P2_FINAL_V2"]

TEMPLATE = {
    "config.json": {"_template": "Populated by the trainer from protocol.json. MUST match the frozen protocol or validate_candidates.py ABORTS."},
    "training_curve.csv": "step,wall_time,iter_time,loss,N_GS,VRAM_gb,psnr,ssim,lpips\n",
    "timing.json": {"_template": "Populated at run end.",
                    "total_wall_s": None, "iterations_per_s": None, "mean_iter_ms": None,
                    "median_iter_ms": None, "forward_ms": None, "backward_ms": None,
                    "nested_fb_ms": None, "optimizer_ms": None, "loss_ms": None},
    "quality.json": {"_template": "Populated at final eval (30000-step, all cameras).",
                     "psnr": None, "ssim": None, "lpips": None, "n_eval_cameras": None},
    "memory.json": {"_template": "Populated at run end.", "peak_vram_gb": None},
    "checkpoints.json": {"_template": "Checkpoint schedule + n_gaussians per checkpoint (for TTQ).", "checkpoints": []},
    "final_status.json": {"_template": "Final run status.", "status": "NOT_RUN",
                          "status_vocabulary": ["SUCCESS", "OOM", "CUDA_ERROR", "NUMERIC_FAILURE", "QUALITY_FAILURE", "INTERRUPTED"]},
    "provenance.json": {"_template": "Populated by capture_provenance.py."},
    "gpu_snapshot.json": {"_template": "Populated by capture_provenance.py (pre-run GPU contamination)."},
}


def init(root: Path):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    scenes = [s["scene_id"] for s in manifest["scenes"]]
    created = 0
    for cand in CANDIDATES:
        for scene in scenes:
            d = root / cand / scene
            d.mkdir(parents=True, exist_ok=True)
            for fname, content in TEMPLATE.items():
                p = d / fname
                if p.exists():
                    continue
                if fname == "training_curve.csv":
                    p.write_text(content)
                else:
                    if isinstance(content, str):
                        p.write_text(content)
                    else:
                        if cand == "P2_FINAL_V2" and fname == "final_status.json":
                            content = dict(content)
                            content["status"] = "PENDING_P2"
                        json.dump(content, p.open("w"), indent=2)
                created += 1
    print(f"created {created} template files under {root}")
    print(f"candidates: {CANDIDATES}; scenes: {len(scenes)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(FINAL30K))
    args = ap.parse_args()
    init(Path(args.root))

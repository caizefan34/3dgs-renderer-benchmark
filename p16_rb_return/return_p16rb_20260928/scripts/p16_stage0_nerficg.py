#!/usr/bin/env python3
"""P16-RB Stage-0 preflight for the NeRFICG-based methods (FasterGS / FasterGSFused).

Runs a short-iteration copy of the common config (same dataset, same recipe; only
NUM_ITERATIONS and FINAL_CHECKPOINT are reduced), then audits against the framework's
own test render:
  - dataset loads; training loop runs to completion
  - native test-set render succeeds
  - split identity + GT identity proven by pixel hashing (p16_audit_nerficg_gt.py)
  - resolution is 1920x1080
  - single GPU only
Writes artifacts/P16_STAGE0_<method>_<scene>.json.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

D = Path("/root/epic05_p16rb")
NERFICG = D / "src" / "nerficg"
CFG = D / "configs"
ART = D / "artifacts"
ENV = "p16_fastergs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["FasterGS", "FasterGSFused"])
    ap.add_argument("--scene", default="room")
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=100)
    args = ap.parse_args()

    base = yaml.safe_load((CFG / f"p16_{args.method.lower()}_{args.scene}_1920.yaml").read_text())
    cfg = json.loads(json.dumps(base))
    model_name = f"p16_preflight_{args.method.lower()}_{args.scene}"
    cfg["TRAINING"]["MODEL_NAME"] = model_name
    cfg["TRAINING"]["NUM_ITERATIONS"] = args.iterations
    cfg["GLOBAL"]["GPU_INDICES"] = [args.gpu]
    cfg["TRAINING"]["BACKUP"]["RENDER_TESTSET"] = True
    # NOTE: BACKUP.FINAL_CHECKPOINT stays at the author value (true) -- the framework's
    # final-checkpoint callback is also what renders the test set; disabling it in a
    # preflight would silently skip the render audit.
    cfg_path = CFG / f"{model_name}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)  # framework resolves GLOBAL.GPU_INDICES natively
    env["PATH"] = f"/root/miniforge3/envs/{ENV}/bin:{env.get('PATH','')}"
    log = D / "logs" / f"stage0_{args.method}_{args.scene}.log"
    t0 = time.perf_counter()
    p = subprocess.run([f"/root/miniforge3/envs/{ENV}/bin/python", "scripts/train.py", "-c", str(cfg_path)],
                       cwd=NERFICG, env=env, text=True, capture_output=True, timeout=3600)
    wall = time.perf_counter() - t0
    log.write_text(f"COMMAND: train.py -c {cfg_path}\n\nSTDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}",
                   encoding="utf-8")
    text = p.stdout + "\n" + p.stderr

    # locate the timestamped output directory the framework creates
    cands = sorted((NERFICG / "output" / args.method).glob(f"{model_name}_*"),
                   key=lambda q: q.stat().st_mtime)
    outdir = cands[-1] if cands else None

    audit = None
    if outdir is not None:
        a = subprocess.run([f"/root/miniforge3/envs/{ENV}/bin/python",
                            str(D / "scripts" / "p16_audit_nerficg_gt.py"),
                            "--outdir", str(outdir), "--scene", args.scene],
                           text=True, capture_output=True, timeout=1800)
        (D / "logs" / f"stage0_{args.method}_{args.scene}_gtaudit.log").write_text(
            a.stdout + "\n" + a.stderr, encoding="utf-8")
        try:
            audit = json.loads(a.stdout)
        except Exception:
            audit = {"status": "FAIL", "raw_tail": a.stdout[-500:]}

    import re
    gpus_used = re.findall(r"using GPU\(s\): (\[[^\]]*\])", text)
    iterations_done = "training: 100%|" in text
    checks = {
        "train_exit_zero": p.returncode == 0,
        "dataset_loaded": "loading dataset" in text,
        "training_loop_completed": iterations_done,
        "timings_txt": bool(outdir and (outdir / "timings.txt").exists()),
        "native_render_ok": bool(outdir and list(outdir.glob("test_*/rgb/*.png"))),
        "split_and_gt_identity_pixels": bool(audit and audit.get("split_identity_by_pixels")),
        "resolution_1920x1080": bool(audit and audit.get("resolution_ok")),
        "single_gpu": all(g.strip() == f"[{args.gpu}]" for g in gpus_used) if gpus_used else False,
    }
    report = {
        "method": args.method, "scene": args.scene, "stage": "STAGE0_PREFLIGHT",
        "config": str(cfg_path), "num_iterations": args.iterations, "gpu": args.gpu,
        "returncode": p.returncode, "wall_s": wall, "log": str(log),
        "output_dir": str(outdir) if outdir else None,
        "checks": checks, "gt_audit": audit,
        "gpus_reported_by_framework": gpus_used,
        "preflight_only_overrides": {"TRAINING.NUM_ITERATIONS": args.iterations},
    }
    report["status"] = "PASS" if all(checks.values()) else "FAIL"
    ART.mkdir(parents=True, exist_ok=True)
    (ART / f"P16_STAGE0_{args.method}_{args.scene}.json").write_text(json.dumps(report, indent=2),
                                                                     encoding="utf-8")
    print(json.dumps({"method": args.method, "status": report["status"], "checks": checks}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

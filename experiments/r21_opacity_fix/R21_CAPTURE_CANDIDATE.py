#!/usr/bin/env python3
"""R21 P4 candidate trajectory capture. Status: IMPLEMENTED; MX NOT_EXECUTED locally.
Adapted from R21_CAPTURE_FIRST_DIVERGENCE.py, SHA256 ca3a230b486cc1daf194d498d9fc8a11ec2c2bf41fe81a85dc0b739ba092c04a.

Reuses the actual R20 training source and GaussianModel with detached callbacks.
Default: hashes every step, complete snapshots at 1/100/599/600, resumable
checkpoints each 100 steps. Use --capture-every-step on a narrowed rerun window.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DIAG = REPO / "experiments/r21_densification"
sys.path.insert(0, str(DIAG))


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--project", type=Path, required=True, help="Existing frozen std root or validated R20 HiGS root")
    ap.add_argument("--backend", required=True, choices=("std_ll", "higs"))
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", type=Path, required=True, help="New directory, never overwrite a run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iterations", type=int, default=30000, help="Original recipe horizon; --stop-after controls diagnostic stop")
    ap.add_argument("--stop-after", type=int, default=600)
    ap.add_argument("--capture-at", default="1,100,599,600")
    ap.add_argument("--capture-every-step", action="store_true")
    ap.add_argument("--resume-every", type=int, default=100)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--static-check", action="store_true")
    # Diagnostic-only arguments used by the causal orchestrator.
    ap.add_argument("--arm", choices=tuple("ABCDEF"))
    ap.add_argument("--donor", type=Path)
    ap.add_argument("--history-reference", type=Path)
    ap.add_argument("--causal", action="store_true")
    return ap


def prepare(args):
    from R21_STATE import sha256, source_sha256
    from R21_INSTRUMENT import instrument_trainer
    trainer = args.repo / "experiments/r20_true_absgrad/R20_TRAINER.py"
    model = args.repo / "baseline/reference_v1/gaussian_model.py"
    manifest = json.loads((DIAG / "R21_SOURCE_IDENTITY.json").read_text(encoding="utf-8"))
    for relative in manifest["required_source_hashes"]:
        actual = args.repo / relative
        if not actual.is_file() or source_sha256(actual) != manifest["required_source_hashes"][relative]:
            raise ValueError(f"Audited source changed; re-audit before capture: {relative}")
    source = instrument_trainer(trainer.read_text(encoding="utf-8"), args.repo)
    ast.parse(source)
    identity = {"trainer_sha256": sha256(trainer), "model_sha256": sha256(model),
                "source_hashes": manifest["required_source_hashes"],
                "source_byte_hashes": {rel:sha256(args.repo/rel) for rel in manifest["required_source_hashes"]},
                "tool_hashes": {p.name:sha256(p) for p in HERE.glob("*.py")},
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()}
    return trainer, source, identity


def main():
    args = parser().parse_args()
    if args.resume_every < 1 or args.stop_after < 1 or args.stop_after > args.iterations:
        raise ValueError("Invalid checkpoint frequency or stop/horizon")
    if args.arm in ("C", "D", "E", "F") and not (args.causal and args.donor and args.resume):
        raise ValueError("Interventions require causal mode, donor and identical input checkpoint")
    if args.arm and ((args.arm == "A") != (args.backend == "std_ll")):
        raise ValueError("Arm A requires std_ll; arms B-F require HiGS")
    trainer, source, identity = prepare(args)
    if args.static_check:
        print(json.dumps({"status":"PASS", "scope":"source anchors and Python AST", **identity}, indent=2))
        return
    if args.out.exists():
        raise FileExistsError(f"Use a NEW output directory: {args.out}")
    args.out.mkdir(parents=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HIGS_PROJECT"] = str(args.project.resolve())
    # Torch is imported by prepare() but no CUDA calls are made before visibility is set.
    import torch
    from R21_STATE import write_json, sha256
    if not torch.cuda.is_available():
        write_json(args.out / "RUN_STATUS.json", {"status":"NOT_EXECUTED", "reason":"CUDA unavailable"})
        raise SystemExit(2)
    identity.update({"torch": str(torch.__version__), "cuda": torch.version.cuda,
                     "device": torch.cuda.get_device_name(0),
                     "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
                     "cudnn_benchmark":torch.backends.cudnn.benchmark,
                     "cudnn_deterministic":torch.backends.cudnn.deterministic,
                     "cudnn_allow_tf32":torch.backends.cudnn.allow_tf32,
                     "matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32,
                     "environment":{key:value for key,value in os.environ.items() if key.startswith(("HIGS_","GSPLAT_"))
                         or key in ("CUDA_VISIBLE_DEVICES","CUBLAS_WORKSPACE_CONFIG","PYTORCH_CUDA_ALLOC_CONF")}})
    # Export the actual missing frozen projection/rendering sources as evidence.
    exported = {}
    rels = ["gsplat/rendering.py", "gsplat/cuda/_wrapper.py", "gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu",
            "gsplat/cuda/include/Utils.cuh", "gsplat/cuda/include/Common.h",
            "gsplat/experimental/render/functional/gaussian_inference.py",
            "gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu",
            "gsplat/csrc.so", "gsplat/experimental/render/kernels/csrc.so",
            "gsplat/cuda/csrc/Rendering.cpp"]
    for rel in rels:
        path = args.project / rel
        if not path.is_file():
            raise FileNotFoundError(f"Actual renderer source/binary required: {path}")
        exported[rel] = sha256(path)
        if path.suffix != ".so":
            target = args.out / "source_evidence" / rel
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
    identity["renderer_hashes"] = exported
    if args.backend == "higs":
        from R21_PREPARE_CANDIDATE import verify_candidate
        identity["r21_candidate"] = verify_candidate(args.project)
    options = {k:(str(v.resolve()) if isinstance(v, Path) else v) for k,v in vars(args).items()}
    options["capture_at"] = [int(v) for v in args.capture_at.split(",") if v]
    options["arm"] = args.arm or ("A" if args.backend == "std_ll" else "B")
    options["identity"] = identity
    os.environ["R21_OPTIONS"] = json.dumps(options)
    (args.out / "instrumented_trainer.py").write_text(source, encoding="utf-8")
    from R21_INSTRUMENT import Session
    sys.argv = [str(trainer), "--backend", args.backend, "--scene", args.scene,
                "--iterations", str(args.iterations), "--outdir", str(args.out),
                "--gpu", "0", "--seed", str(args.seed), "--densify-mode", "absgrad"]
    namespace = {"__name__":"__main__", "__file__":str(trainer), "Session":Session}
    try:
        exec(compile(source, str(args.out / "instrumented_trainer.py"), "exec"), namespace)
    except BaseException as exc:
        write_json(args.out / "RUN_STATUS.json", {"status":"FAIL", "exception":repr(exc)})
        raise
    finally:
        if any(sha256(args.project / rel) != expected for rel,expected in exported.items()):
            raise RuntimeError("Renderer source/binary changed during capture")


if __name__ == "__main__":
    main()

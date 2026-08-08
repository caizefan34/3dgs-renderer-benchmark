#!/usr/bin/env python
"""Run one audited from-SfM job for the HiGS paper protocol."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higs_training_commands import (  # noqa: E402
    build_training_invocation,
    trainer_cfg_kwargs,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, help="method id from the protocol")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "benchmark" / "higs-paper-protocol.json",
        help="explicit protocol JSON (must match the invocation builder)",
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--init-type", choices=("sfm",), default="sfm")
    parser.add_argument("--max-steps", type=int, default=30_000,
                        help="training budget in steps (1..30000; audited per-method protocol override)")
    parser.add_argument(
        "--smoke-steps",
        type=int,
        help="Run a non-paper smoke test; its metadata is never paper-eligible.",
    )
    return parser.parse_args()


def _load_trainer(source_dir: Path):
    source_dir = source_dir.resolve()
    trainer_path = source_dir / "examples" / "simple_trainer.py"
    if not trainer_path.is_file():
        raise RuntimeError(f"missing pinned trainer: {trainer_path}")
    sys.path.insert(0, str(source_dir / "examples"))
    sys.path.insert(0, str(source_dir))
    spec = importlib.util.spec_from_file_location("paper_gsplat_simple_trainer", trainer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trainer: {trainer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    imported = Path(sys.modules["gsplat"].__file__).resolve()
    if source_dir not in imported.parents:
        raise RuntimeError(
            f"gsplat resolved outside pinned source tree: {imported}"
        )
    return module, imported


def main() -> int:
    args = _parse_args()
    if args.data_dir.suffix.lower() in {".ckpt", ".ply", ".pt", ".pth"}:
        raise SystemExit("--data-dir must be a COLMAP dataset directory")
    if not args.data_dir.is_dir():
        raise SystemExit(f"dataset directory does not exist: {args.data_dir}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if args.method not in protocol.get("methods", {}):
        raise SystemExit(
            f"method {args.method!r} is not defined in protocol {args.protocol}"
        )
    invocation = build_training_invocation(
        protocol=protocol,
        method=args.method,
        scene=args.scene,
        seed=args.seed,
        data_dir=args.data_dir,
        result_dir=args.result_dir,
        source_dir=args.source_dir,
        python_executable=sys.executable,
        repository_root=ROOT,
    )

    args.result_dir.mkdir(parents=True, exist_ok=True)
    extension_cache = (
        ROOT
        / "artifacts"
        / "cuda-build"
        / f"{args.source_dir.resolve().name}-py{sys.version_info.major}{sys.version_info.minor}"
    )
    extension_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(extension_cache))
    trainer, imported_gsplat = _load_trainer(args.source_dir)
    if args.smoke_steps is not None and args.smoke_steps < 1:
        raise SystemExit("--smoke-steps must be positive")
    if not (1 <= args.max_steps <= 30_000):
        raise SystemExit("--max-steps must be in [1, 30000]")
    effective_steps = args.smoke_steps or args.max_steps

    upstream_set_seed = trainer.set_random_seed
    trainer.set_random_seed = lambda _upstream_default: upstream_set_seed(args.seed)
    method_spec = protocol["methods"][args.method]
    algorithm = method_spec.get("algorithm", {})
    cfg_kwargs = trainer_cfg_kwargs(method_spec)
    cfg = trainer.Config(
        disable_viewer=True,
        data_dir=str(args.data_dir.resolve()),
        result_dir=str(args.result_dir.resolve()),
        init_type=args.init_type,
        max_steps=effective_steps,
        eval_steps=sorted({min(s, effective_steps) for s in (7_000, 15_000, 30_000)}),
        save_steps=sorted({min(s, effective_steps) for s in (7_000, 15_000, 30_000)}),
        disable_video=True,
        **cfg_kwargs,
    )

    started_at = datetime.now(timezone.utc).isoformat()
    import torch

    torch.cuda.synchronize()
    started = time.perf_counter()
    trainer.main(0, 0, 1, cfg)
    torch.cuda.synchronize()
    wall_time = time.perf_counter() - started
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    artifact_files = sorted(
        list(args.result_dir.glob("ckpts/*.pt"))
        + list(args.result_dir.glob("stats/*.json"))
        + list(args.result_dir.glob("cfg.yml"))
    )
    artifacts = [
        {
            "path": path.relative_to(args.result_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in artifact_files
    ]
    dataset_inventory = args.data_dir / "dataset_inventory.json"
    metadata = {
        "schema_version": "1.0",
        "method": args.method,
        "scene": args.scene,
        "run_kind": "smoke" if args.smoke_steps is not None else "paper",
        "paper_eligible": args.smoke_steps is None,
        "initialization": "from_scratch_sfm",
        "iterations": effective_steps,
        "seed": args.seed,
        "timing_boundary": "dataset_ready_to_final_checkpoint",
        "started_at_utc": started_at,
        "wall_time_seconds": wall_time,
        "source_dir": str(args.source_dir.resolve()),
        "imported_gsplat": str(imported_gsplat),
        "torch_extensions_dir": os.environ["TORCH_EXTENSIONS_DIR"],
        "source": invocation["source"],
        "source_audit": invocation["source_audit"],
        "algorithm": algorithm,
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "hardware": {
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
        },
        "dataset": {
            "path": str(args.data_dir.resolve()),
            "inventory_sha256": (
                _sha256(dataset_inventory) if dataset_inventory.is_file() else None
            ),
        },
        "artifacts": artifacts,
        "clean_process": True,
    }
    (args.result_dir / "paper-run-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


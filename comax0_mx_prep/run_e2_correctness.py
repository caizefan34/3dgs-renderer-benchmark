#!/usr/bin/env python
"""run_e2_correctness - E2-prep CORRECTNESS anchor (numerics only, NO timing).

Fixes the watch_and_run.sh E2-prep gap: the watcher launched the dense harness
directly without a guard proof, so require_guarded_process() raised immediately
(silent). It also completes the sys.path setup the dense harness omits (prod src/
for comax_prod_ext) and makes the CPU-written harnesses GPU-safe.

The trusted oracle (correctness_prod_harness._reference_scan) builds its pixel grid
on CPU and subtracts means2d, so the fixture is intentionally LEFT ON CPU: the
comax_scan oracle therefore runs byte-identical to the validated CPU selftest.
Only the CUDA kernel entrypoints receive device copies of the inputs. This keeps
the CUDA path from being its own oracle while leaving the reference untouched.

Launched via scripts/run_on_mx.py, which writes the GUARD.json proof (runner_pid +
gpu_uuid + CUDA_VISIBLE_DEVICES) that require_guarded_process() demands.

Correctness-only: NO wall-clock, throughput, or bandwidth claims. Establishes the
CUDA fused dense + envelope kernels vs the trusted comax_scan oracle anchor that
the E2 gate depends on. Does NOT advance the scientific verdict and does NOT run
any E2 timing.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROD = os.environ.get(
    "COMAX0_PROD_ROOT",
    "/mnt/storage_pool/liaoyuanjun/comax0_mx/COMAX0_CUDA_PROD",
)
PKG_SCRIPTS = os.environ.get(
    "COMAX0_PACKAGE_SCRIPTS",
    "/mnt/storage_pool/liaoyuanjun/comax0_mx/package/COMAX0_MX_DSH/scripts",
)


def _bootstrap_paths():
    for p in (
        os.path.join(PROD, "harness"),
        os.path.join(PROD, "src"),
        os.path.join(PROD, "logs", "torch_ext", "comax_prod"),
        PKG_SCRIPTS,
    ):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    os.environ["COMAX0_PROD_ROOT"] = PROD
    os.environ["COMAX0_PACKAGE_SCRIPTS"] = PKG_SCRIPTS
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(PROD, "logs", "torch_ext"))


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _install_device_fixups(ph, eh):
    """GPU-enable ONLY the CUDA kernel entrypoints; leave the CPU oracle untouched.

    The kernels read tensor data_ptrs directly with no device check and require
    CUDA float32/int32 inputs. We move the CPU fixture tensors onto CUDA at the
    kernel boundary. The comax_scan oracle keeps running on CPU (fixture not moved)
    so it stays byte-identical to the validated CPU selftest.
    """
    import torch
    import comax_prod_ext as _cpe
    import comax_envelope_ext as _cee

    def _cuda(t, dtype=None):
        if not torch.is_tensor(t):
            return t
        return t.to("cuda", dtype) if dtype is not None else t.to("cuda")

    orig_dense = _cpe.comax_render_dense

    def _dense(means2d, conic, opacity, color, groups, tile_ids, tile_offsets,
               width, height, tw, th, background=None):
        means2d = _cuda(means2d)
        conic = _cuda(conic)
        opacity = _cuda(opacity)
        color = _cuda(color)
        groups = _cuda(groups, torch.int32)
        tile_ids = _cuda(tile_ids, torch.int32)
        tile_offsets = _cuda(tile_offsets, torch.int32)
        background = (torch.zeros(3, dtype=means2d.dtype, device="cuda")
                      if background is None else _cuda(background, means2d.dtype))
        return orig_dense(means2d, conic, opacity, color, groups, tile_ids, tile_offsets,
                          width, height, tw, th, background)

    _cpe.comax_render_dense = _dense

    orig_build = _cee.comax_build_envelope

    def _build(means2d, conic, opacity, color, depth, groups, tile_ids, tile_offsets,
               n_tiles, tw, th, width, height, background=None):
        means2d = _cuda(means2d)
        conic = _cuda(conic)
        opacity = _cuda(opacity)
        color = _cuda(color)
        depth = _cuda(depth, means2d.dtype)
        groups = _cuda(groups, torch.int32)
        tile_ids = _cuda(tile_ids, torch.int32)
        tile_offsets = _cuda(tile_offsets, torch.int32)
        background = (torch.zeros(3, dtype=means2d.dtype, device="cuda")
                      if background is None else _cuda(background, means2d.dtype))
        return orig_build(means2d, conic, opacity, color, depth, groups, tile_ids,
                          tile_offsets, n_tiles, tw, th, width, height, background)

    _cee.comax_build_envelope = _build


def main():
    _bootstrap_paths()
    import torch  # noqa: F401  (ensures the CUDA torch stack is importable)
    import correctness_prod_harness as ph
    import correctness_envelope_harness as eh
    ph._setup_env()
    eh._setup_env()
    _install_device_fixups(ph, eh)
    from mx_guard import require_guarded_process

    cfg, proof = require_guarded_process()  # guard BEFORE any heavy work

    out_dir = os.environ.get("COMAX0_OUTPUT_DIR", ".")
    dense_out = os.path.join(out_dir, "COMAX_CUDA_CORRECTNESS.json")
    env_out = os.path.join(out_dir, "COMAX_CUDA_CORRECTNESS_ENVELOPE.json")

    dense_err = env_err = None
    dense_checks = dense_counters = env_checks = env_counters = None
    try:
        dense_checks, dense_counters = ph.run_gpu()
    except Exception as exc:  # noqa: BLE001
        dense_err = repr(exc)
    try:
        env_checks, env_counters = eh.run()
    except Exception as exc:  # noqa: BLE001
        env_err = repr(exc)

    dense_all = bool(dense_checks) and all(c.get("pass", False) for c in dense_checks)
    env_all = bool(env_checks) and all(c.get("pass", False) for c in env_checks)

    common = {
        "schema": "comax0.cuda_correctness.e2prep.v1",
        "run_id": os.environ.get("COMAX0_RUN_ID"),
        "ts_utc": _now(),
        "stage": "e2_correctness_prep",
        "host": cfg.get("expected_hostname"),
        "gpu_uuid": proof.get("gpu_uuid"),
        "runner_pid": proof.get("runner_pid"),
        "tolerance_rgb_abs_max": ph.PREREG_ABS,
        "performance_evidence_valid": False,
        "timing_run": False,
        "shared_gpu_numerics_only": False,
        "note": (
            "E2-prep correctness anchor only. Establishes fused CUDA dense + "
            "envelope kernels vs the trusted comax_scan oracle. NO timing, "
            "throughput, or bandwidth claim. Does NOT advance the master verdict."
        ),
    }
    dense_payload = dict(common, mode="dense", checks=dense_checks,
                         counters_last=(dense_counters.tolist() if dense_counters is not None else None),
                         error=dense_err, all_pass=dense_all)
    env_payload = dict(common, mode="envelope", checks=env_checks,
                       counters=(env_counters if env_counters is not None else None),
                       error=env_err, all_pass=env_all)
    _write(dense_out, dense_payload)
    _write(env_out, env_payload)

    overall = {
        "schema": "comax0.e2_correctness_summary.v1",
        "ts_utc": _now(),
        "run_id": os.environ.get("COMAX0_RUN_ID"),
        "gpu_uuid": proof.get("gpu_uuid"),
        "dense_all_pass": dense_all and dense_err is None,
        "envelope_all_pass": env_all and env_err is None,
        "dense_error": dense_err,
        "envelope_error": env_err,
        "performance_evidence_valid": False,
        "timing_run": False,
        "note": "Correctness anchor for the E2 gate. A pass here is NECESSARY, not SUFFICIENT, for E2.",
    }
    overall["all_pass"] = overall["dense_all_pass"] and overall["envelope_all_pass"]
    summary_path = os.path.join(out_dir, "E2_CORRECTNESS_SUMMARY.json")
    _write(summary_path, overall)
    print(json.dumps(overall, indent=2))
    return 0 if overall["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

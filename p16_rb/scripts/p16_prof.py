"""P16-RB bottleneck census: timing-only profiling runtime (instrumentation payload).

Discipline (prereg step 11):
  - this module plus small edits are injected into a COPIED source tree; the applied
    edits are stored as a unified diff next to the raw traces
  - behavior-inert unless P16_PROF=1: no profiler, no spans, no RNG effect
  - camera-id + loss CSV logging is active whenever P16_PROF_OUT is set (both the
    control and the instrumented leg of the A/B neutrality check use the same build)
  - optional diagnostic seed (P16_PROF_SEED) is applied ONLY to the A/B control pair

Environment:
  P16_PROF=1|0        enable profiler windows + spans (default 0)
  P16_PROF_OUT=<dir>  output directory (CSV + trace_*.json + meta.json)
  P16_PROF_N=<int>    NATIVE iteration count of the replayed recipe (window math)
  P16_PROF_BASE=0|1   first loop index (nerficg 0, graphdeco 1)
  P16_PROF_W=<int>    window length in iterations (default 20)
  P16_PROF_METHOD / P16_PROF_SCENE / P16_PROF_SEED   labels
"""
import atexit
import contextlib
import csv
import json
import os
import platform
import socket
import time
from pathlib import Path

import torch

_W = int(os.environ.get("P16_PROF_W", "20"))
_N = int(os.environ.get("P16_PROF_N", "0") or 0)
_BASE = int(os.environ.get("P16_PROF_BASE", "0") or 0)


class _Span:
    __slots__ = ("ctx",)

    def __init__(self, ctx):
        self.ctx = ctx

    def __enter__(self):
        if self.ctx is not None:
            self.ctx.__enter__()
        return self

    def __exit__(self, *exc):
        if self.ctx is not None:
            self.ctx.__exit__(*exc)
        return False


class P16:
    enabled = os.environ.get("P16_PROF", "0") == "1"
    outdir = os.environ.get("P16_PROF_OUT", "")
    method = os.environ.get("P16_PROF_METHOD", "?")
    scene = os.environ.get("P16_PROF_SCENE", "?")
    seed = os.environ.get("P16_PROF_SEED", "")
    native_n = _N
    base = _BASE
    w = _W

    _t0 = None
    _prof = None
    _stage = None
    _windows = {}
    _counts = {}
    _csv_f = None
    _csv_w = None
    _iter_ctx = None
    _iter_idx = None

    @classmethod
    def init(cls):
        cls._t0 = time.time()
        atexit.register(cls.close)
        if cls.seed != "":
            import random

            import numpy as np

            s = int(cls.seed)
            random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)
            torch.cuda.manual_seed_all(s)
        if cls.outdir:
            Path(cls.outdir).mkdir(parents=True, exist_ok=True)
            cls._csv_f = open(Path(cls.outdir) / "iterations.csv", "w", newline="", encoding="utf-8")
            cls._csv_w = csv.writer(cls._csv_f)
            cls._csv_w.writerow(["iteration", "cameras", "loss"])
            cls._csv_f.flush()
        if not cls.enabled:
            return
        n = cls.native_n
        if n > 2 * cls.w + 5:
            lo = int(0.6 * n) - cls.w // 2
            cls._windows = {
                "mid": (lo, lo + cls.w),
                "final": (n - 1 - cls.w, n - 1),  # excludes the terminal iteration (save/test may fire there)
            }
        cls._counts = {k: 0 for k in cls._windows}

    @classmethod
    def log_iter(cls, iteration, cameras, loss):
        if cls._csv_w is None:
            return
        try:
            v = repr(float(loss))
        except Exception:
            v = str(loss)
        cls._csv_w.writerow([iteration, cameras, v])
        cls._csv_f.flush()

    @classmethod
    def _close_iter(cls):
        if cls._iter_ctx is not None:
            ctx, cls._iter_ctx, cls._iter_idx = cls._iter_ctx, None, None
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass

    @classmethod
    def iter_begin(cls, iteration):
        """Call as the first statement of every training-loop body iteration."""
        if not cls.enabled:
            return
        idx = iteration - cls.base
        cls._close_iter()
        if cls._prof is not None:
            _, hi = cls._windows[cls._stage]
            if idx >= hi:
                cls._stop_prof()
        if cls._prof is None:
            for stage, (lo, hi) in cls._windows.items():
                if lo <= idx < hi:
                    cls._start_prof(stage)
                    break
        for stage, (lo, hi) in cls._windows.items():
            if lo <= idx < hi:
                cls._counts[stage] += 1
                break
        if cls._prof is not None:
            try:
                cls._iter_ctx = torch.profiler.record_function(f"P16.iter {idx}")
                cls._iter_ctx.__enter__()
                cls._iter_idx = idx
            except Exception:
                cls._iter_ctx = None

    @classmethod
    def span(cls, name):
        if cls._prof is not None:
            try:
                return _Span(torch.profiler.record_function("P16." + name))
            except Exception:
                return _Span(None)
        return _Span(None)

    @classmethod
    def _start_prof(cls, stage):
        cls._stage = stage
        try:
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,
                with_stack=False,
                profile_memory=False,
            )
            prof.start()
        except Exception as e:
            cls._prof = None
            cls._stage = None
            (Path(cls.outdir) / f"trace_{stage}.ERROR.txt").write_text(repr(e), encoding="utf-8")
            return
        cls._prof = prof

    @classmethod
    def _stop_prof(cls):
        stage = cls._stage
        cls._close_iter()
        prof, cls._prof, cls._stage = cls._prof, None, None
        if prof is None:
            return
        try:
            prof.stop()
            prof.export_chrome_trace(str(Path(cls.outdir) / f"trace_{stage}.json"))
        except Exception as e:
            (Path(cls.outdir) / f"trace_{stage}.ERROR.txt").write_text(repr(e), encoding="utf-8")

    @classmethod
    def close(cls):
        cls._close_iter()
        if cls._prof is not None:
            cls._stop_prof()
        if cls._csv_f is not None:
            cls._csv_f.close()
            cls._csv_f = None
            cls._csv_w = None
        if cls.enabled and cls.outdir:
            meta = {
                "method": cls.method,
                "scene": cls.scene,
                "enabled": cls.enabled,
                "native_iterations": cls.native_n,
                "base": cls.base,
                "window_len": cls.w,
                "seed": cls.seed,
                "windows_idx": {k: list(v) for k, v in cls._windows.items()},
                "windows_iterations_recorded": dict(cls._counts),
                "csv": str(Path(cls.outdir) / "iterations.csv"),
                "hostname": socket.gethostname(),
                "cwd": os.getcwd(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(cls._t0)) if cls._t0 else None,
                "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            (Path(cls.outdir) / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


@contextlib.contextmanager
def _noop():
    yield

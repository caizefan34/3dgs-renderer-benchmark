"""VOS trace driver — Phase 4 visibility-trace collection on the frozen C0.

ZERO source diff on the frozen trainer: this wrapper imports
publication_trainer_p13x.py as a module, wraps its bootstrap_c0() so that
immediately after bootstrap we patch gsplat.experimental.rasterize_gaussian_higs_dynamic
with a capturing wrapper, then calls the trainer's main() verbatim.

Capture semantics (metadata-only, mirroring the D0 trace-neutrality rules):
- One grad-enabled render call per training iteration (evaluate() runs under
  torch.no_grad -> excluded).
- Per call: bitmap = (densification_info["radii"][0] > 0).any(-1)  [N] bool
  -- exactly the trainer's own visibility_filter definition, and exactly the
  culling visible_ids set (radii are scattered into a full-N zeros tensor).
- Bitmap packed with torch.packbits on GPU, copied to host (synchronous --
  the trainer already synchronizes once per iteration), queued to a writer
  thread that appends [iter, N, nbytes, sh_degree] + packed bytes to a .bin.
- No RNG calls added, no input tensor mutated, returned dict untouched.

Output files (in --outdir):
- vos_trace.bin          raw per-iteration packed bitmaps
- vos_trace_meta.json    index + provenance (written at exit)
- (all normal trainer outputs: results.json, ckpt_*.pt, ...)

Usage:
  python vos_trace_driver.py --scene room --iterations 30000 --gpu 0 \
      --outdir /root/vos_traces/room_T30 --trainer /root/epic05_p13x/publication_trainer_p13x.py
Extra passthrough args after -- are forwarded to the trainer (e.g. --loss fused).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import queue
import struct
import sys
import threading
import time

import torch


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class TraceRecorder:
    """Captures per-iteration packed visibility bitmaps; writer thread flushes."""

    def __init__(self, bin_path, meta):
        self.bin_path = bin_path
        self.meta = meta
        self.q = queue.Queue(maxsize=64)
        self.records = []
        self.call_idx = 0
        self.t_first = None
        self.t_last = None
        self.stop = False
        self.f = open(bin_path, "wb", buffering=1 << 20)
        self.written_bytes = 0
        self.thread = threading.Thread(target=self._writer, daemon=True)
        self.thread.start()
        # packbits convention self-test — DEFERRED: performed lazily on the first
        # record() call, on the training tensors' device. A probe here would
        # initialize CUDA BEFORE the trainer sets CUDA_VISIBLE_DEVICES, silently
        # defeating GPU pinning (observed: all runs landed on GPU 0).
        self._pack_w = None
        self.meta["packbits_probe_note"] = (
            "probe byte recorded lazily at first capture; MSB-first packer: "
            "element i -> bit (7 - i%8) of byte i//8")

    def _pack_bits(self, bitmap):
        """Pack a [N] bool tensor to uint8[ceil(N/8)], MSB-first, on its device."""
        n = bitmap.shape[0]
        pad = (-n) % 8
        b = bitmap.to(torch.uint8)
        if pad:
            b = torch.cat([b, torch.zeros(pad, dtype=torch.uint8, device=b.device)])
        if self._pack_w is None or self._pack_w.device != b.device:
            self._pack_w = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1],
                                        device=b.device, dtype=torch.uint8)
        return (b.reshape(-1, 8) * self._pack_w).sum(dim=1).to(torch.uint8)

    def record(self, out, kw):
        radii = out["densification_info"]["radii"]  # [1, N, 2]
        r = radii[0]                                # [N, 2]
        bitmap = (r > 0).any(dim=-1)                # [N] bool
        N = int(r.shape[0])
        packed = self._pack_bits(bitmap)            # uint8 [ceil(N/8)] on device
        if self.meta.get("packbits_probe_byte") is None:
            probe = torch.tensor([True, False, False, False, False,
                                  False, False, False], device=bitmap.device)
            self.meta["packbits_probe_byte"] = int(self._pack_bits(probe)[0].item())
            self.meta["probe_device"] = str(bitmap.device)
        host = packed.cpu()                         # sync copy (trainer syncs anyway)
        nbytes = int(packed.numel())
        it = self.call_idx + 1
        shd = kw.get("sh_degree")
        shd = -1 if shd is None else int(shd)
        if self.t_first is None:
            self.t_first = time.time()
        self.t_last = time.time()
        self.records.append((it, N, nbytes, shd))
        self.q.put((it, N, nbytes, shd, host.numpy().tobytes()))
        self.call_idx += 1

    def _writer(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            it, N, nbytes, shd, blob = item
            self.f.write(struct.pack("<4i", it, N, nbytes, shd))
            self.f.write(blob)
            self.written_bytes += 4 * 4 + len(blob)

    def close(self):
        self.q.put(None)
        self.thread.join(timeout=30)
        self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--trainer", default="/root/epic05_p13x/publication_trainer_p13x.py")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--final-eval", default="subset")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra args forwarded verbatim to the trainer")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    bin_path = os.path.join(args.outdir, "vos_trace.bin")
    meta_path = os.path.join(args.outdir, "vos_trace_meta.json")

    meta = {
        "scene": args.scene,
        "iterations": args.iterations,
        "gpu": args.gpu,
        "seed": args.seed if args.seed is not None else 42,
        "trainer_path": args.trainer,
        "trainer_sha256": sha256_file(args.trainer),
        "torch": torch.__version__,
        "capture_semantics": ("bitmap=(densification_info.radii[0]>0).any(-1); grad-enabled "
                              "calls only (training iterations); metadata-only, no RNG, no "
                              "tensor mutation"),
        "record_format": "little-endian int32 x4 [iter, N, nbytes, sh_degree] then nbytes "
                         "packed-bit bytes (torch.packbits of the [N] bool bitmap)",
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if torch.cuda.is_available():
        meta["gpu_name"] = torch.cuda.get_device_name(0)
        try:
            meta["gpu_uuid"] = str(torch.cuda.get_device_properties(0).uuid)
        except Exception:
            meta["gpu_uuid"] = "unavailable"

    recorder = TraceRecorder(bin_path, meta)

    # ---- import the frozen trainer module (no source modification) ---------
    spec = importlib.util.spec_from_file_location("vos_trainer", args.trainer)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)

    # ---- wrap bootstrap_c0 to patch the render entry right after bootstrap -
    orig_bootstrap = T.bootstrap_c0

    def bootstrap_wrapped():
        ident = orig_bootstrap()
        import gsplat.experimental as E
        real_fn = E.rasterize_gaussian_higs_dynamic

        def capturing_render(means, quats, scales, opacities, colors, *a, **kw):
            out = real_fn(means, quats, scales, opacities, colors, *a, **kw)
            try:
                if torch.is_grad_enabled():
                    recorder.record(out, kw)
            except Exception as e:  # NEVER kill training because of tracing
                meta.setdefault("record_errors", []).append(repr(e))
            return out

        E.rasterize_gaussian_higs_dynamic = capturing_render
        meta["patched_module"] = "gsplat.experimental.rasterize_gaussian_higs_dynamic"
        return ident

    T.bootstrap_c0 = bootstrap_wrapped

    # ---- build the trainer's argv and run its main() verbatim --------------
    targv = [args.trainer, "--arm", "c0", "--scene", args.scene,
             "--loss", "fused", "--gpu", str(args.gpu), "--eps2d", "0.3",
             "--iterations", str(args.iterations), "--checkpoints",
             "--outdir", args.outdir,
             "--final-eval", args.final_eval, "--timing-grade", "FUNCTIONAL_ONLY"]
    if args.seed is not None:
        targv += ["--seed", str(args.seed)]
    if not args.no_lpips:
        targv += ["--lpips"]
    targv += list(args.extra)
    sys.argv = targv

    t0 = time.time()
    exit_ok = True
    try:
        T.main()
    except SystemExit as e:
        exit_ok = (e.code in (0, None))
    except BaseException:
        exit_ok = False
        raise
    finally:
        recorder.close()
        meta["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta["driver_wall_s"] = time.time() - t0
        meta["n_captured_training_iters"] = recorder.call_idx
        meta["trace_bin_bytes"] = recorder.written_bytes
        meta["capture_span_s"] = ((recorder.t_last - recorder.t_first)
                                  if (recorder.t_first and recorder.t_last) else None)
        meta["record_errors"] = meta.get("record_errors", [])
        # compact index: one entry per iteration [iter, N, nbytes, sh_degree]
        meta["records"] = [list(r) for r in recorder.records]
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        expected = args.iterations
        if recorder.call_idx != expected:
            meta["WARNING"] = (f"captured {recorder.call_idx} grad-enabled calls, "
                               f"expected {expected}")
            with open(meta_path, "w") as f:
                json.dump(meta, f)
    sys.exit(0 if exit_ok else 1)


if __name__ == "__main__":
    main()

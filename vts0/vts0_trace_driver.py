"""VTS0 trace driver — zero-diff topology + visibility trace on frozen C0.

VTS0 question: does topology-aware dependency scheduling provide ADDITIONAL safe
optimizer skipping beyond visibility-based VOS?

This driver imports publication_trainer_p13x.py as a module (ZERO source diff on
the frozen trainer / gaussian_model) and, immediately after bootstrap, wraps:
  1. gsplat.experimental.rasterize_gaussian_higs_dynamic  -> per-iteration
     visibility bitmap (radii>0).any(-1), exactly the trainer visibility_filter.
  2. gaussian_model.GaussianModel.densify_and_prune        -> per-event
     clone/split/prune counts (topology-forced + lineage + settlement ratios).

Capture is metadata-only: no RNG calls, no input tensor mutation, returned dict
untouched. Trace-neutrality: the trainer already synchronizes once per iter.

Output (in --outdir):
  vts0_trace.bin        per-iter packed visibility bitmaps
  vts0_topology.json    per-topology-event records
  vts0_meta.json        index + provenance
Usage:
  python vts0_trace_driver.py --scene room --iterations 3000 --gpu 0 \
      --outdir /root/vts0_traces/room_dyn --trainer /root/epic05_p13x/publication_trainer_p13x.py
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
        self._pack_w = None
        self.meta["packbits_probe_note"] = (
            "element i -> bit (7 - i%8) of byte i//8; MSB-first")

    def _pack_bits(self, bitmap):
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
        radii = out["densification_info"]["radii"]
        r = radii[0]
        bitmap = (r > 0).any(dim=-1)
        N = int(r.shape[0])
        packed = self._pack_bits(bitmap)
        if self.meta.get("packbits_probe_byte") is None:
            probe = torch.tensor([True, False, False, False,
                                  False, False, False, False], device=bitmap.device)
            self.meta["packbits_probe_byte"] = int(self._pack_bits(probe)[0].item())
            self.meta["probe_device"] = str(bitmap.device)
        host = packed.cpu()
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
    ap.add_argument("--iterations", type=int, default=3000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--trainer", default="/root/epic05_p13x/publication_trainer_p13x.py")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--final-eval", default="subset")
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    bin_path = os.path.join(args.outdir, "vts0_trace.bin")
    topo_path = os.path.join(args.outdir, "vts0_topology.json")
    meta_path = os.path.join(args.outdir, "vts0_meta.json")

    meta = {
        "scene": args.scene,
        "iterations": args.iterations,
        "gpu": args.gpu,
        "seed": args.seed if args.seed is not None else 42,
        "trainer_path": args.trainer,
        "trainer_sha256": sha256_file(args.trainer),
        "torch": torch.__version__,
        "capture_semantics": ("visibility bitmap=(densification_info.radii[0]>0).any(-1); "
                              "topology events via densify_and_prune return counts; "
                              "metadata-only, no RNG, no tensor mutation"),
        "record_format": "int32 x4 [iter,N,nbytes,sh_degree] + packed-bit bytes",
        "start_utc": time.strftime("%Y-%dT%H:%dT%SZ", time.gmtime()),
        "topology_events": [],
    }
    if torch.cuda.is_available():
        meta["gpu_name"] = torch.cuda.get_device_name(0)
        try:
            meta["gpu_uuid"] = str(torch.cuda.get_device_properties(0).uuid)
        except Exception:
            meta["gpu_uuid"] = "unavailable"

    recorder = TraceRecorder(bin_path, meta)

    spec = importlib.util.spec_from_file_location("vts0_trainer", args.trainer)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)

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
            except Exception as e:
                meta.setdefault("record_errors", []).append(repr(e))
            return out

        E.rasterize_gaussian_higs_dynamic = capturing_render

        # Wrap densify_and_prune on the already-imported gaussian_model class.
        import gaussian_model as GM
        real_dap = GM.GaussianModel.densify_and_prune

        def capturing_dap(self, max_grad, min_opacity, extent, max_screen_size, radii):
            n_before = int(self._xyz.shape[0])
            out = real_dap(self, max_grad, min_opacity, extent, max_screen_size, radii)
            n_after = int(self._xyz.shape[0])
            ev = {
                "iter": recorder.call_idx + 1,
                "n_before": n_before,
                "n_after": n_after,
                "cloned": int(out["cloned"]),
                "split": int(out["split"]),
                "pruned_total": int(out["pruned_total"]),
                "pruned_opacity": int(out["pruned_opacity"]),
                "pruned_screen": int(out["pruned_screen"]),
                "pruned_world": int(out["pruned_world"]),
                "new_children": int(out["cloned"]) + 2 * int(out["split"]),
            }
            meta["topology_events"].append(ev)
            return out

        GM.GaussianModel.densify_and_prune = capturing_dap
        meta["patched_modules"] = [
            "gsplat.experimental.rasterize_gaussian_higs_dynamic",
            "gaussian_model.GaussianModel.densify_and_prune",
        ]
        return ident

    T.bootstrap_c0 = bootstrap_wrapped

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
        meta["end_utc"] = time.strftime("%Y-%dT%H:%dT%SZ", time.gmtime())
        meta["driver_wall_s"] = time.time() - t0
        meta["n_captured_training_iters"] = recorder.call_idx
        meta["trace_bin_bytes"] = recorder.written_bytes
        meta["capture_span_s"] = ((recorder.t_last - recorder.t_first)
                                  if (recorder.t_first and recorder.t_last) else None)
        meta["record_errors"] = meta.get("record_errors", [])
        meta["records"] = [list(r) for r in recorder.records]
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        with open(topo_path, "w") as f:
            json.dump(meta["topology_events"], f, indent=2)
        expected = args.iterations
        if recorder.call_idx != expected:
            meta["WARNING"] = (f"captured {recorder.call_idx} grad-enabled calls, "
                               f"expected {expected}")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
    sys.exit(0 if exit_ok else 1)


if __name__ == "__main__":
    main()


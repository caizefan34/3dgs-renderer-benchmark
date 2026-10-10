#!/usr/bin/env python3
"""R17 interceptor for the *unmodified* gsplat isect_tiles CUDA op.

Used by P3 (attribution cells), P4 (benchmark) and P5 (4.41M safety gate).

The interceptor never changes kernel behavior:
- the full original call is executed and its outputs are returned unchanged;
- an extra diagnostic call is executed on a seeded subset of the *same*
  already-projected tensors (identical to R15/R16 semantics, seed 42, k=1000).

Modes:
  stats           full call + subset rerun + optional tile coverage (P3)
  record_only     full call + call table, no coverage, no subset (P4)
  estimate_only   subset rerun only, memory projection, then SafeAbort (P5 phase 1)
  gated_full      subset estimate, memory gate; run full only if safely fitting (P5 phase 2)
  capture_only    subset rerun + save + SystemExit-equivalent (R15 reproduction)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from common import choose_ids, np_stats, sha256_file, write_json, EXPECTED


class SafeAbort(Exception):
    """Raised when the recorder intentionally stops execution before a risky full call."""

    def __init__(self, info: dict):
        super().__init__("R17_SAFE_ABORT")
        self.info = info


class IsectRecorder:
    def __init__(self, orig, mode: str = "stats", *, samples: int = None, seed: int = None,
                 estimate_samples: int = 20000, estimate_seed: int = 123,
                 save_dir: Path | None = None, tag: str = "cell",
                 do_coverage: bool | None = None, memory_margin_bytes: int = 4 << 30,
                 log=print):
        self.orig = orig
        self.mode = mode
        self.samples = samples if samples is not None else EXPECTED["sample_k"]
        self.seed = seed if seed is not None else EXPECTED["sample_seed"]
        self.estimate_samples = estimate_samples
        self.estimate_seed = estimate_seed
        self.save_dir = Path(save_dir) if save_dir else None
        self.tag = tag
        if do_coverage is None:
            do_coverage = (mode == "stats")
        self.do_coverage = do_coverage
        self.memory_margin_bytes = memory_margin_bytes
        self.log = log
        self.qualifying_calls: list[dict] = []
        self.first_capture: dict | None = None
        self.estimate: dict | None = None
        self.aborted = False
        self.started = time.time()

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _is_qualifying(args, kwargs) -> bool:
        return (len(args) >= 6 and kwargs.get("conics") is not None
                and kwargs.get("opacities") is not None)

    def _subset_rerun(self, args, kwargs, k: int, seed: int):
        means2d, radii, depths, tile_size, tile_width, tile_height = args[:6]
        conics = kwargs.get("conics")
        opacities = kwargs.get("opacities")
        mask = kwargs.get("tile_mask")
        if kwargs.get("packed", False):
            raise RuntimeError("R17 recorder requires HiGS packed=False")
        full_n = int(means2d.shape[-2])
        ids = choose_ids(full_n, k, seed)
        index = torch.as_tensor(ids, device=means2d.device, dtype=torch.long)
        small = {
            "means2d": means2d.index_select(-2, index).contiguous(),
            "radii": radii.index_select(-2, index).contiguous(),
            "depths": depths.index_select(-1, index).contiguous(),
            "conics": conics.index_select(-2, index).contiguous(),
            "opacities": opacities.index_select(-1, index).contiguous(),
        }
        gpu_kwargs = dict(kwargs)
        gpu_kwargs["sort"] = False
        gpu_kwargs["segmented"] = False
        gpu_kwargs["conics"] = small["conics"]
        gpu_kwargs["opacities"] = small["opacities"]
        counts, gpu_ids, gpu_flatten = self.orig(
            small["means2d"], small["radii"], small["depths"],
            int(tile_size), int(tile_width), int(tile_height), **gpu_kwargs)
        torch.cuda.synchronize(means2d.device)
        total = int(counts.to(torch.int64).sum().item())
        emitted = int(gpu_ids.numel())
        meta = {
            "source": "R17 interceptor => gsplat.cuda._wrapper.isect_tiles",
            "full_projected_gaussian_count": full_n,
            "sample_gaussian_count": int(index.numel()),
            "sample_seed": seed,
            "projected_tensor_shape": list(means2d.shape),
            "sampled_tensor_shape": list(small["means2d"].shape),
            "width_in_tiles": int(tile_width),
            "height_in_tiles": int(tile_height),
            "tile_size": int(tile_size),
            "n_images": int(small["means2d"].numel() // (index.numel() * 2)),
            "gpu_count_sum": total,
            "gpu_emitted_intersection_count": emitted,
            "gpu_count_and_emit_agree": bool(total == emitted == int(gpu_flatten.numel())),
            "has_conics": True,
            "has_opacities": True,
            "has_tile_mask": mask is not None,
            "torch_version": torch.__version__,
            "r17_mode": self.mode,
            "r17_tag": self.tag,
        }
        pkg = {
            "means2d": small["means2d"].detach().cpu().contiguous(),
            "radii": small["radii"].detach().cpu().contiguous(),
            "depths": small["depths"].detach().cpu().contiguous(),
            "conics": small["conics"].detach().cpu().contiguous(),
            "opacities": small["opacities"].detach().cpu().contiguous(),
            "tile_mask": mask.detach().cpu().contiguous() if mask is not None else None,
            "gpu_counts": counts.detach().cpu().contiguous(),
            "sample_indices": torch.as_tensor(ids, dtype=torch.int64),
            "meta": meta,
        }
        return pkg

    def _save_capture(self, pkg: dict, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pkg, path)
        summary = dict(pkg["meta"])
        summary["capture_path"] = str(path)
        summary["capture_sha256"] = sha256_file(path)
        summary["per_primitive_max_gpu_count"] = int(pkg["gpu_counts"].max().item()) if pkg["gpu_counts"].numel() else 0
        summary["per_primitive_image_tile_bound"] = int(
            pkg["meta"]["width_in_tiles"] * pkg["meta"]["height_in_tiles"])
        summary["above_tile_bound_count"] = int(
            (pkg["gpu_counts"] > summary["per_primitive_image_tile_bound"]).sum().item())
        write_json(path.with_suffix(".json"), summary)
        return summary

    def _coverage(self, isect_ids: torch.Tensor, n_images: int, wt: int, ht: int) -> dict:
        """Unique (image,tile) coverage from the real full isect_ids, chunked to bound temp memory.

        Key layout (gsplat csrc): key = (iid << (32+tile_n_bits)) | (tile_id << 32) | depth_bits,
        tile_n_bits = (n_tiles-1).bit_length() (mirrors bits_for_count in MathUtils.h).
        """
        n_tiles = wt * ht
        bits = (n_tiles - 1).bit_length() if n_tiles > 1 else 0
        tmask = (1 << bits) - 1
        cov = torch.zeros(n_images * n_tiles, dtype=torch.bool, device=isect_ids.device)
        step = 64 << 20
        n = int(isect_ids.numel())
        max_flat = -1
        for i in range(0, n, step):
            ik = (isect_ids[i:i + step] >> 32).to(torch.int64)
            tile_id = ik & tmask
            iid = ik >> bits
            flat = iid * n_tiles + tile_id
            if flat.numel():
                max_flat = max(max_flat, int(flat.max().item()))
            cov[flat] = True
        if max_flat >= n_images * n_tiles:
            raise RuntimeError(
                f"Coverage decode overflow: max flat {max_flat} >= {n_images * n_tiles}")
        per_image = cov.view(n_images, n_tiles)
        per_cam = [float(v.float().mean().item()) for v in per_image]
        return {
            "n_images": n_images,
            "tiles_per_image": n_tiles,
            "tile_n_bits": bits,
            "tiles_hit_total": int(cov.sum().item()),
            "coverage_frac_total": float(cov.float().mean().item()),
            "coverage_frac_per_camera": per_cam,
        }

    def _estimate_and_gate(self, pkg: dict, full_n: int, args) -> dict:
        counts = pkg["gpu_counts"].to(torch.int64).numpy()
        md = pkg["meta"]
        n_images = int(md["n_images"])
        n_tiles = md["width_in_tiles"] * md["height_in_tiles"]
        mean_c = float(counts.mean()) if counts.size else 0.0
        est_isects = mean_c * full_n * n_images
        p90_c = float(np.quantile(counts, 0.9)) if counts.size else 0.0
        est_bytes_lower = est_isects * 12.0                      # isect_ids int64 + flatten int32, one buffer set
        est_bytes_documented = est_bytes_lower * 2.0 * 1.5      # count+emit passes and sort slack (documented conservative)
        free, total = torch.cuda.mem_get_info()
        state_bytes = int(torch.cuda.memory_allocated())
        info = {
            "estimate_samples": int(counts.size),
            "estimate_seed": self.estimate_seed,
            "call_full_n": int(full_n),
            "call_n_images": n_images,
            "call_total_primitives": int(full_n * n_images),
            "tiles_per_image": int(n_tiles),
            "sample_count_stats": np_stats(counts, prefix="count_"),
            "sample_frac_at_tile_bound": float((counts >= n_tiles).mean()) if counts.size else None,
            "est_isects_mean_scaled": est_isects,
            "est_isects_p90_scaled": p90_c * full_n * n_images,
            "projected_bytes_lower": float(est_bytes_lower),
            "projected_bytes_documented_conservative": float(est_bytes_documented),
            "gpu_free_bytes": int(free),
            "gpu_total_bytes": int(total),
            "gpu_allocated_bytes_at_gate": state_bytes,
            "memory_margin_bytes": int(self.memory_margin_bytes),
            "note": ("Estimate = per-primitive tile counts from a seeded subset of the same real "
                     "projected tensors, scaled by call size (full_n x n_images). Byte projection "
                     "is an engineering projection (int64 isect ids + int32 flatten ids, two "
                     "passes, x1.5 sort slack), NOT a measurement of any allocation site."),
        }
        safe = (est_bytes_documented + self.memory_margin_bytes) < free
        info["decision"] = "RUN_FULL" if safe else "SKIP_FULL_INSUFFICIENT_HEADROOM"
        info["headroom_after_projection_bytes"] = float(free - est_bytes_documented - self.memory_margin_bytes)
        self.estimate = info
        write_json((self.save_dir / f"R17_{self.tag}_SAFETY_GATE.json") if self.save_dir
                   else Path(f"R17_{self.tag}_SAFETY_GATE.json"), info)
        return info

    # -- main entry ----------------------------------------------------------
    def __call__(self, *args, **kwargs):
        if not self._is_qualifying(args, kwargs):
            return self.orig(*args, **kwargs)

        means2d, radii, depths, tile_size, tile_width, tile_height = args[:6]
        full_n = int(means2d.shape[-2])
        n_images = int(means2d.numel() // (full_n * 2))
        call_idx = len(self.qualifying_calls)
        record = {
            "call_index": call_idx,
            "means2d_shape": list(means2d.shape),
            "tile_grid": [int(tile_width), int(tile_height)],
            "n_images": n_images,
        }

        if call_idx == 0:
            if self.mode in ("stats", "capture_only", "estimate_only", "gated_full"):
                pkg = self._subset_rerun(
                    args, kwargs,
                    self.samples if self.mode != "estimate_only" else self.estimate_samples,
                    self.seed if self.mode != "estimate_only" else self.estimate_seed)
                record["subset"] = pkg["meta"]
                self.first_capture = {"pkg": pkg, "full_n": full_n}
                if self.save_dir is not None and self.mode != "estimate_only":
                    self._save_capture(pkg, self.save_dir / f"{self.tag}_r15_style_capture.pt")

            if self.mode == "capture_only":
                self.qualifying_calls.append(record)
                self.aborted = True
                raise SafeAbort({"mode": "capture_only", "record": record})

            if self.mode == "estimate_only":
                info = self._estimate_and_gate(pkg, full_n, args)
                record["estimate"] = info
                self.qualifying_calls.append(record)
                self.aborted = True
                raise SafeAbort({"mode": "estimate_only", "record": record, "estimate": info})

            if self.mode == "gated_full":
                info = self._estimate_and_gate(pkg, full_n, args)
                record["estimate"] = info
                if info["decision"] != "RUN_FULL":
                    self.qualifying_calls.append(record)
                    self.aborted = True
                    raise SafeAbort({"mode": "gated_full_skipped", "record": record, "estimate": info})

        # full original call (unmodified op, original tensors)
        t0 = time.time()
        out = self.orig(*args, **kwargs)
        torch.cuda.synchronize(means2d.device)
        record["full_call_seconds"] = time.time() - t0
        counts = out[0]
        record["sum_counts"] = int(counts.to(torch.int64).sum().item())
        record["sum_counts_over_tiles"] = record["sum_counts"] / float(tile_width * tile_height * n_images)
        record["emitted_ids"] = int(out[1].numel())
        record["count_emit_agree"] = bool(record["sum_counts"] == record["emitted_ids"])
        if call_idx == 0 and self.do_coverage:
            record["coverage"] = self._coverage(out[1], n_images, int(tile_width), int(tile_height))
            counts_cpu = counts.detach().cpu().numpy()
            record["counts_stats"] = np_stats(counts_cpu, prefix="count_")
            record["frac_counts_gt_tile_bound"] = (
                float((counts_cpu > int(tile_width) * int(tile_height)).mean())
                if counts_cpu.size else None)
        self.qualifying_calls.append(record)
        return out

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "tag": self.tag,
            "qualifying_calls": self.qualifying_calls,
            "n_qualifying_calls": len(self.qualifying_calls),
            "first_call_records": self.qualifying_calls[0] if self.qualifying_calls else None,
            "estimate": self.estimate,
            "aborted": self.aborted,
            "seconds": time.time() - self.started,
        }


def install(recorder: IsectRecorder):
    import gsplat.cuda._wrapper as w
    orig = w.isect_tiles
    recorder.orig = orig
    w.isect_tiles = recorder
    return orig


def uninstall(orig):
    import gsplat.cuda._wrapper as w
    w.isect_tiles = orig

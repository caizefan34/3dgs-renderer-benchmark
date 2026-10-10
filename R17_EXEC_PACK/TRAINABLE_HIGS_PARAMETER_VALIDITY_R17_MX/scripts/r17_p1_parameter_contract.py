#!/usr/bin/env python3
"""R17-P1: parameter input contract for the HiGS frozen entry + real checkpoint distributions.

Evidence chain (no reinterpretation):
  1. What the training stack stores in checkpoints      (capture() field list)
  2. What the training stack feeds the renderer         (getters + render_with_meta)
  3. What the HiGS renderer entry documents as contract (docstrings, frozen tree)
  4. What the AccuTile CUDA kernel actually computes    (verbatim source lines)
  5. What the actual bicycle checkpoint contains        (distributions, CPU)
  6. Branch arithmetic of the CUDA t-formula on the real raw vs activated opacities

Outputs: R17_PARAMETER_CONTRACT.md, R17_PARAMETER_DISTRIBUTIONS.json
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CKPT, EXPECTED, OUT, PROJECT, ensure_out_dirs, load_checkpoint,
                    np_stats, read_common_constants, sha256_file, write_json)

TRAIN_REPO_CANDIDATES = [
    str(Path.home() / "3dgs-renderer-benchmark"),
    "/home/liaoyuanjun/3dgs-renderer-benchmark",
]


def excerpt(path: Path, needle: str, before: int = 6, after: int = 6):
    if not path.is_file():
        return {"found": False, "file": str(path), "needle": needle}
    lines = path.read_text(errors="replace").splitlines()
    hits = [i for i, l in enumerate(lines) if needle in l]
    if not hits:
        return {"found": False, "file": str(path), "needle": needle}
    i = hits[0]
    lo, hi = max(0, i - before), min(len(lines), i + after + 1)
    return {
        "found": True, "file": str(path), "needle": needle,
        "line_of_needle": i + 1,
        "lines": [f"{n+1}: {lines[n]}" for n in range(lo, hi)],
        "file_sha256": sha256_file(path),
    }


def find_train_repo():
    for c in TRAIN_REPO_CANDIDATES:
        if Path(c).is_dir():
            return Path(c)
    return None


def t_branch_table(opacity_np: np.ndarray, alpha: float) -> dict:
    """Exact classification against the CUDA t-formula branches."""
    o = opacity_np.astype(np.float64)
    n = o.size
    le0 = int((o <= 0).sum())
    in_skip = int(((o > 0) & (o <= alpha)).sum())
    cap_cut = alpha * math.exp(0.5)
    mid = int(((o > alpha) & (o < cap_cut)).sum())
    cap = int((o >= cap_cut).sum())
    return {
        "n": int(n),
        "opacity_le_0_logf_of_negative": le0,
        "opacity_in_(0,alpha]_t_would_be_negative": in_skip,
        "opacity_in_(alpha,alpha*e^0.5)_t_in_(0,1)": mid,
        "opacity_ge_alpha*e^0.5_t_saturates_at_1": cap,
        "frac_with_no_real_threshold_behavior": (le0 + in_skip) / n if n else None,
    }


def main():
    ensure_out_dirs()
    dist = {
        "deliverable": "R17_PARAMETER_DISTRIBUTIONS.json",
        "checkpoint_expected": EXPECTED,
        "project": str(PROJECT),
        "status": "NOT_EXECUTED",
        "source_evidence": {},
        "distribution_stats": {},
        "branch_arithmetic": {},
    }
    ckpt, ck_info = load_checkpoint(CKPT, require=False)
    dist["checkpoint_observed"] = ck_info
    if ckpt is None:
        write_json(OUT / "R17_PARAMETER_DISTRIBUTIONS.json", dist)
        raise SystemExit("Checkpoint missing — P1 NOT_EXECUTED.")

    # ---- distributions -----------------------------------------------------
    raw_op = ckpt["opacity"].float().reshape(-1)
    raw_sc = ckpt["scaling"].float()
    raw_q = ckpt["rotation"].float()
    shs = ckpt["shs"].float()

    sig = torch.sigmoid(raw_op)
    exp_sc = torch.exp(raw_sc)
    qn = torch.linalg.norm(raw_q, dim=-1)

    stats = {
        "opacity_raw_logit": {
            **np_stats(raw_op.numpy(), ""),
            "count_le_0": int((raw_op <= 0).sum()),
            "count_le_alpha": None,  # filled below
            "count_outside_unit_interval": int(((raw_op < 0) | (raw_op > 1)).sum()),
            "min": float(raw_op.min()), "max": float(raw_op.max()),
        },
        "opacity_sigmoid": {
            **np_stats(sig.numpy(), ""),
        },
        "scaling_raw_log_scale": {
            **np_stats(raw_sc.numpy(), ""),
            "count_le_0": int((raw_sc <= 0).sum()),
            "count_negative": int((raw_sc < 0).sum()),
            "count_nonfinite": int((~torch.isfinite(raw_sc)).sum()),
        },
        "scaling_exp_activated": {
            **np_stats(exp_sc.numpy(), ""),
            "count_nonfinite": int((~torch.isfinite(exp_sc)).sum()),
        },
        "rotation_raw": {
            "norm_mean": float(qn.mean()), "norm_min": float(qn.min()),
            "norm_max": float(qn.max()),
            "frac_norm_dev_gt_1e-3": float((torch.abs(qn - 1.0) > 1e-3).float().mean()),
            "frac_norm_dev_gt_1e-2": float((torch.abs(qn - 1.0) > 1e-2).float().mean()),
        },
        "shs": {
            "shape": list(shs.shape), "dtype": str(ckpt["shs"].dtype),
            "min": float(shs.min()), "max": float(shs.max()),
            "count_nonfinite": int((~torch.isfinite(shs)).sum()),
        },
        "shapes": {
            "xyz": list(ckpt["xyz"].shape),
            "opacity": list(ckpt["opacity"].shape),
            "scaling": list(ckpt["scaling"].shape),
            "rotation": list(ckpt["rotation"].shape),
            "shs": list(ckpt["shs"].shape),
        },
    }

    const = read_common_constants()
    dist["constants"] = const
    alpha = const.get("alpha_threshold")
    if alpha:
        stats["opacity_raw_logit"]["count_le_alpha"] = int((raw_op <= alpha).sum())
        stats["opacity_sigmoid"]["count_le_alpha"] = int((sig <= alpha).sum())
        stats["opacity_sigmoid"]["count_le_0_00392"] = int((sig <= 1.0 / 255).sum())

    dist["distribution_stats"] = stats

    # ---- branch arithmetic on the FULL checkpoint --------------------------
    if alpha:
        dist["branch_arithmetic"] = {
            "cuda_formula": "float t = fminf(GAUSSIAN_EXTEND*GAUSSIAN_EXTEND, 2.0f*__logf(opacity / ALPHA_THRESHOLD));",
            "alpha_threshold": alpha,
            "gaussian_extend": const.get("gaussian_extend"),
            "raw_logit_used_as_opacity": t_branch_table(raw_op.numpy(), alpha),
            "sigmoid_activated": t_branch_table(sig.numpy(), alpha),
        }

    # ---- source evidence ---------------------------------------------------
    gif = PROJECT / "gsplat/experimental/render/functional/gaussian_inference.py"
    cu = PROJECT / "gsplat/cuda/csrc/IntersectTile.cu"
    header = PROJECT / "gsplat/cuda/include/Common.h"
    wrapper = PROJECT / "gsplat/cuda/_wrapper.py"
    ev = {
        "frozen_entry_docstring": excerpt(gif, "rasterize_gaussian_higs_frozen(", 0, 40),
        "trainable_contract_docstring": excerpt(gif, "scales: Scales (activated, positive)", 5, 5),
        "trainable_backward_note": excerpt(gif, "Stage A uses the standard gsplat rasterization", 2, 4),
        "accutile_cuda_t_line": excerpt(cu, "__logf(opacity / ALPHA_THRESHOLD)", 12, 12),
        "accutile_skip_radii_zero": excerpt(cu, "if(radius_x <= 0 || radius_y <= 0)", 0, 8),
        "common_alpha_threshold": excerpt(header, "ALPHA_THRESHOLD", 2, 2),
        "common_gaussian_extend": excerpt(header, "GAUSSIAN_EXTEND", 2, 2),
        "wrapper_isect_docstring": excerpt(wrapper, "conservative ellipse intersection", 2, 4),
    }
    repo = find_train_repo()
    if repo:
        gm = repo / "baseline/reference_v1/gaussian_model.py"
        tr = repo / "baseline/reference_v1/trainer.py"
        ev["trainer_getters"] = excerpt(gm, "def get_opacity", 0, 30)
        ev["trainer_render_activation"] = excerpt(tr, "def render_with_meta", 0, 18)
        ev["trainer_capture_fields"] = excerpt(gm, "def capture", 0, 25)
        ev["train_repo"] = str(repo)
    else:
        ev["train_repo"] = "NOT_AVAILABLE_ON_MX"
    dist["source_evidence"] = {k: (v if isinstance(v, str) else
                                   {kk: vv for kk, vv in v.items() if kk != "lines"})
                               for k, v in ev.items()}
    dist["status"] = "EXECUTED"

    # ---- contract markdown -------------------------------------------------
    def fmt(x):
        return json.dumps(x, indent=2, default=str)

    op = stats["opacity_raw_logit"]
    sig_st = stats["opacity_sigmoid"]
    sc = stats["scaling_raw_log_scale"]
    rot = stats["rotation_raw"]
    verdicts = {}
    verdicts["opacity"] = (
        "VIOLATED" if op["count_outside_unit_interval"] > 0.5 * op["n"] else "UNKNOWN")
    verdicts["scaling"] = "VIOLATED" if sc["count_negative"] > 0 else "UNKNOWN"
    verdicts["rotation"] = (
        "VIOLATED" if rot["frac_norm_dev_gt_1e-3"] > 0.01 else "UNKNOWN")

    def block(title, ex):
        if not isinstance(ex, dict) or not ex.get("found"):
            return f"### {title}\n\nNOT_FOUND / NOT_AVAILABLE\n"
        return ("### {}\n\n`{}` (sha256 `{}`, line {})\n\n```\n{}\n```\n").format(
            title, ex["file"], str(ex.get("file_sha256"))[:16], ex.get("line_of_needle"),
            "\n".join(ex["lines"]))

    md = f"""# R17 Parameter Input Contract — evidence report

Status: **{dist['status']}**

Checkpoint: `{ck_info['path']}`
sha256 `{ck_info['sha256']}` — N = {ck_info.get('n_gaussians'):,}

## 1. What the training stack stores (RAW parameters)

Checkpoint fields come from `gaussian_model.capture()`. Stored values are the
optimizer variables `_scaling`, `_rotation`, `_opacity` — i.e. log-scales,
(approximately unit) quaternion vectors, and logit opacities.

{block('trainer_capture_fields', ev.get('trainer_capture_fields'))}

## 2. What the training stack feeds the renderer (ACTIVATED parameters)

`render_with_meta` passes the *getter* outputs: sigmoid(opacity), exp(scaling),
normalize(rotation). The renderer never sees raw logits in the real training loop.

{block('trainer_getters', ev.get('trainer_getters'))}

{block('trainer_render_activation', ev.get('trainer_render_activation'))}

## 3. What the HiGS renderer documents as its contract (frozen tree)

tRAINABLE entry docstring (the frozen entry forwards to the same native path):

{block('trainable_contract_docstring', ev.get('trainable_contract_docstring'))}

Backward note:

{block('trainable_backward_note', ev.get('trainable_backward_note'))}

Frozen entry:

{block('frozen_entry_docstring', ev.get('frozen_entry_docstring'))}

## 4. What the AccuTile CUDA kernel actually computes

{block('accutile_cuda_t_line', ev.get('accutile_cuda_t_line'))}

Threshold constants in the frozen tree:

{block('common_alpha_threshold', ev.get('common_alpha_threshold'))}

{block('common_gaussian_extend', ev.get('common_gaussian_extend'))}

## 5. Real checkpoint distributions (CPU, whole tensor)

### Opacity (stored raw logits)
- min {op['min']}, median {op['p50']}, p99 {op['p99']}, max {op['max']}
- values outside [0,1]: **{op['count_outside_unit_interval']:,} / {op['n']:,}**
- logits <= 0 (logf of negative in the t-formula): **{op['count_le_0']:,}**
- logits <= ALPHA_THRESHOLD: {op.get('count_le_alpha')}
- after sigmoid: median {sig_st['p50']}, p99 {sig_st['p99']}, max {sig_st['max']};
  count <= ALPHA_THRESHOLD: **{sig_st.get('count_le_alpha')}**

### Scaling (stored log-scales)
- raw: min {sc['min']}, median {sc['p50']}, max {sc['max']},
  negative count **{sc['count_negative']:,}**
- exp(raw): min {stats['scaling_exp_activated']['min']}, median {stats['scaling_exp_activated']['p50']}, max {stats['scaling_exp_activated']['max']}

### Rotation (stored quaternion vectors)
- norm: min {rot['norm_min']}, mean {rot['norm_mean']}, max {rot['norm_max']}
- fraction with |norm - 1| > 1e-3: **{rot['frac_norm_dev_gt_1e-3']:.6f}**

## 6. Branch arithmetic of the CUDA t-formula on the real data

Formula: `t = fminf(1, 2*log(opacity/alpha))` (alpha = {alpha}).

Raw logits fed as "opacity":
{fmt(dist['branch_arithmetic'].get('raw_logit_used_as_opacity'))}

Sigmoid-activated opacities (the contract):
{fmt(dist['branch_arithmetic'].get('sigmoid_activated'))}

Interpretation rule (deterministic, from the source lines above):
- raw logit <= 0 -> `__logf` of a non-positive number produces NaN; `fminf(1, NaN)`
  returns the non-NaN operand 1.0 -> the *maximum* extent branch is taken.
- 0 < raw logit <= alpha -> `log(opacity/alpha) < 0` -> negative t in the ellipse
  math -> sqrt of negative inside `accutile_ellipse_intersection`.
- activated opacity in (alpha, alpha*e^0.5) -> the only branch with a genuine
  threshold behavior.

## 7. Preliminary contract verdicts (final in R17_FINAL_VERDICT.json)

- OPACITY_INPUT_CONTRACT: **{verdicts['opacity']}** (R3/R8/R15/R16 fed raw logits)
- SCALE_INPUT_CONTRACT: **{verdicts['scaling']}** (R3/R8 fed log-scales)
- ROTATION_INPUT_CONTRACT: **{verdicts['rotation']}** (R8 fed unnormalized; R3 normalized)

These verdicts describe the *harness inputs of R3–R16*, not the checkpoint
(which correctly stores raw optimizer variables) and not the training loop
(which activates before rendering).
"""
    (OUT / "R17_PARAMETER_CONTRACT.md").write_text(md, encoding="utf-8")
    write_json(OUT / "R17_PARAMETER_DISTRIBUTIONS.json", dist)
    print(f"[write] {OUT / 'R17_PARAMETER_CONTRACT.md'}", flush=True)
    print(json.dumps({"verdicts": verdicts, "alpha": alpha,
                      "raw_branch": dist['branch_arithmetic'].get('raw_logit_used_as_opacity')},
                     indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()

"""Minimal CPU semantics check for SAD-GS hybrid sparse SH updates.

This models only gradient accumulation, last-camera visibility gating, sparse Adam's
per-Gaussian predicate, and zero_grad(set_to_none=True); it is not CUDA parity.
"""
import json
import math
from copy import deepcopy
from pathlib import Path

BETA1 = 0.9
BETA2 = 0.999
LR = 1e-2
EPS = 1e-10


def sparse_adam_step(params, grads, exp_avg, exp_avg_sq, visible):
    """Mirror adamUpdateCUDA's per-Gaussian if (tiles_touched[g_idx]) body."""
    for g_idx, is_visible in enumerate(visible):
        if not is_visible:
            continue
        for j, grad in enumerate(grads[g_idx]):
            m = BETA1 * exp_avg[g_idx][j] + (1.0 - BETA1) * grad
            v = BETA2 * exp_avg_sq[g_idx][j] + (1.0 - BETA2) * grad * grad
            params[g_idx][j] -= LR * m / (math.sqrt(v) + EPS)
            exp_avg[g_idx][j] = m
            exp_avg_sq[g_idx][j] = v


def main():
    # Two SH parameter rows, each with two coefficients. Rows are independent Gaussians.
    params0 = [[0.25, -0.5], [1.0, 2.0]]
    mom0 = [[0.0, 0.0], [0.0, 0.0]]
    var0 = [[0.0, 0.0], [0.0, 0.0]]
    grads_cam1 = [[1.25, -0.5], [0.5, 0.25]]
    grads_cam2 = [[0.0, 0.0], [-0.2, 0.75]]
    visible_cam1 = [True, True]
    visible_cam2 = [False, True]

    # loss.backward() accumulates into .grad; source does not divide each loss by batch_size.
    accumulated = [[a + b for a, b in zip(row1, row2)]
                   for row1, row2 in zip(grads_cam1, grads_cam2)]
    last_camera_mask = list(visible_cam2)  # train.py overwrites last_radii every batch item.

    actual_params = deepcopy(params0)
    actual_mom = deepcopy(mom0)
    actual_var = deepcopy(var0)
    sparse_adam_step(actual_params, accumulated, actual_mom, actual_var, last_camera_mask)
    grad_after_zero_grad = None  # shoptimizer.zero_grad(set_to_none=True)

    union_mask = [a or b for a, b in zip(visible_cam1, visible_cam2)]
    union_params = deepcopy(params0)
    union_mom = deepcopy(mom0)
    union_var = deepcopy(var0)
    sparse_adam_step(union_params, accumulated, union_mom, union_var, union_mask)

    first_only_unchanged = (
        actual_params[0] == params0[0]
        and actual_mom[0] == mom0[0]
        and actual_var[0] == var0[0]
    )
    control_updated = actual_params[1] != params0[1]
    union_control_updated = union_params[0] != params0[0]
    result = {
        "case": "batch_size=2; Gaussian 0 visible and has SH grad in camera 1 only; Gaussian 1 visible in both",
        "camera_1_visibility": visible_cam1,
        "camera_2_last_visibility": visible_cam2,
        "camera_1_sh_grad": grads_cam1,
        "camera_2_sh_grad": grads_cam2,
        "accumulated_sh_grad_before_step": accumulated,
        "actual_last_mask_sparse_step": {
            "params_before": params0,
            "params_after_step": actual_params,
            "exp_avg_after_step": actual_mom,
            "exp_avg_sq_after_step": actual_var,
            "grad_after_zero_grad_set_to_none": grad_after_zero_grad,
        },
        "union_mask_counterfactual": {
            "visibility": union_mask,
            "params_after_step": union_params,
            "exp_avg_after_step": union_mom,
            "exp_avg_sq_after_step": union_var,
        },
        "checks": {
            "first_camera_only_gaussian_skipped_by_last_mask": first_only_unchanged,
            "its_accumulated_nonzero_gradient_then_cleared": accumulated[0] != [0.0, 0.0] and grad_after_zero_grad is None,
            "last_visible_control_gaussian_updated": control_updated,
            "union_mask_would_update_first_only_gaussian": union_control_updated,
        },
        "all_checks_pass": all((first_only_unchanged, control_updated, union_control_updated,
                                 accumulated[0] != [0.0, 0.0], grad_after_zero_grad is None)),
        "limitations": [
            "pure-Python CPU semantic model; no PyTorch autograd or FP32 rounding",
            "does not execute author CUDA rasterizer or SparseGaussianAdam extension",
            "models the two-camera mask/gradient condition only, not real scene frequency or training impact",
        ],
    }
    out = Path(__file__).with_name("cpu_gate_result.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    assert result["all_checks_pass"]


if __name__ == "__main__":
    main()

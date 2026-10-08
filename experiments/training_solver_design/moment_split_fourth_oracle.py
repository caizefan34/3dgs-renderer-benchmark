"""FP64 1-D density and adjacent alpha oracle for a compensated split.

This is not an alpha-composited 3DGS renderer or a training result.
"""

import json
import math
from pathlib import Path

import numpy as np


OUT = Path(__file__).with_name("moment_split_fourth_oracle.json")


def main():
    x = np.linspace(-8.0, 8.0, 20001, dtype=np.float64)
    dx = float(x[1] - x[0])
    weights = np.full_like(x, dx)
    weights[[0, -1]] *= 0.5
    phi = np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    h2 = (x * x - 1.0) * phi
    h4 = (x**4 - 6.0 * x * x + 3.0) * phi
    old = np.stack([phi, x * phi, h2], axis=1)

    def norm(v):
        return float(np.sqrt(np.sum(weights * v * v)))

    def remove_old(v):
        gram = old.T @ (weights[:, None] * old)
        rhs = old.T @ (weights * v)
        return v - old @ np.linalg.solve(gram, rhs)

    h4_residual = remove_old(h4)
    residual_fraction = norm(h4_residual) / norm(h4)
    rows = []
    q = 0.6
    parent_alpha = q * np.exp(-0.5 * x * x)
    for h in (0.5, 0.35, 0.25, 0.175, 0.125):
        child_sigma = math.sqrt(1.0 - h * h)
        mixture = (
            np.exp(-0.5 * ((x - h) / child_sigma) ** 2)
            + np.exp(-0.5 * ((x + h) / child_sigma) ** 2)
        ) / (2.0 * child_sigma * math.sqrt(2.0 * math.pi))
        actual = mixture - phi
        fourth = -(h**4 / 12.0) * h4
        actual_residual = remove_old(actual)
        fourth_residual = remove_old(fourth)
        child_plus_raw = q * np.exp(-0.5 * ((x - h) / child_sigma) ** 2) / child_sigma
        child_minus_raw = q * np.exp(-0.5 * ((x + h) / child_sigma) ** 2) / child_sigma
        assert np.max(child_plus_raw) < 1.0 and np.max(child_minus_raw) < 1.0
        # A non-Gaussian child kernel exactly preserves parent alpha at h=0.
        pair_alpha = 1.0 - np.sqrt((1.0 - child_plus_raw) * (1.0 - child_minus_raw))
        actual_alpha_change = pair_alpha - parent_alpha
        second_order_alpha = h * h * parent_alpha**2 * x**2 / (2.0 * (1.0 - parent_alpha))
        rows.append({
            "h_over_sigma": h,
            "actual_l2": norm(actual),
            "fourth_approx_relative_l2_error": norm(actual - fourth) / norm(actual),
            "conditioned_actual_l2": norm(actual_residual),
            "conditioned_approx_relative_l2_error": (
                norm(actual_residual - fourth_residual) / norm(actual_residual)
            ),
            "conditioned_direction_cosine": float(
                np.sum(weights * actual_residual * fourth_residual)
                / (norm(actual_residual) * norm(fourth_residual))
            ),
            "transmittance_exact_pair_alpha_delta_l2": norm(actual_alpha_change),
            "alpha_second_order_relative_l2_error": (
                norm(actual_alpha_change - second_order_alpha) / norm(actual_alpha_change)
            ),
        })

    unnormalized_g = np.exp(-0.5 * x * x)
    half_alpha = 0.5 * parent_alpha
    pair_alpha_at_zero_shift = 1.0 - (1.0 - half_alpha) ** 2
    zero_shift_alpha_center_error = float(pair_alpha_at_zero_shift[len(x) // 2] - q)

    # Exact occlusion counterexample: an unrelated colored splat lies between
    # two same-color children even though total family transmittance matches.
    rng = np.random.default_rng(20260927)
    interleaving_max_error = 0.0
    for _ in range(1000):
        a1, a2, b = rng.uniform(0.0, 0.9, size=3)
        c, d, background = rng.uniform(0.0, 1.0, size=(3, 3))
        a = 1.0 - (1.0 - a1) * (1.0 - a2)
        original = a * c + (1.0 - a) * b * d + (1.0 - a) * (1.0 - b) * background
        split = (a1 * c + (1.0 - a1) * b * d
                 + (1.0 - a1) * (1.0 - b) * a2 * c
                 + (1.0 - a1) * (1.0 - b) * (1.0 - a2) * background)
        predicted = (1.0 - a1) * a2 * b * (d - c)
        interleaving_max_error = max(
            interleaving_max_error, float(np.max(np.abs(split - original - predicted)))
        )

    # The density tangent is fourth order after moment compensation, but the
    # ordinary image L2 metric does not make H4 orthogonal to old parameters.
    assert residual_fraction > 0.0 and residual_fraction < 1.0
    assert rows[-1]["fourth_approx_relative_l2_error"] < rows[0]["fourth_approx_relative_l2_error"]
    assert rows[-1]["conditioned_direction_cosine"] > 0.99
    assert rows[-1]["alpha_second_order_relative_l2_error"] < rows[0]["alpha_second_order_relative_l2_error"]
    assert abs(zero_shift_alpha_center_error + 0.09) < 1e-12
    assert interleaving_max_error < 1e-15
    output = {
        "scope": "Normalized additive 1-D density plus adjacent equal-color alpha composition; not a 3DGS renderer or a speed/quality result.",
        "old_tangent_basis": "phi, x*phi, (x^2-1)*phi under plain image L2",
        "h4_energy_after_old_tangent_projection_fraction": residual_fraction,
        "moment_compensated_splits": rows,
        "transmittance_exact_adjacent_pair_alpha_leading_term": "h^2 * a(x)^2 * x^2 / (2*(1-a(x))) at sigma=1; O(h^2), not H4/O(h^4)",
        "ordinary_half_opacity_pair_zero_shift_alpha_center_error_at_q_0_6": zero_shift_alpha_center_error,
        "interleaved_child_color_change_identity": "(1-a1)*a2*b*(d-c) for parent-before-B vs child1-B-child2 with matched total transmittance",
        "interleaved_child_identity_max_fp64_error_1000_random_cases": interleaving_max_error,
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

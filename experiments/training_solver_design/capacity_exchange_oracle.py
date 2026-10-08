"""FP64 compositor checks for delete-then-birth capacity exchange algebra.

This is a scalar/rgb mathematical oracle only; it does not model FastGS's
hard support, early termination, geometry, optimizer, or GPU cost.
"""

import json
from pathlib import Path

import numpy as np


OUT = Path(__file__).with_suffix(".json")


def composite(alphas, colors, background):
    value = np.asarray(background, dtype=np.float64).copy()
    for alpha, color in reversed(list(zip(alphas, colors))):
        value = alpha * color + (1.0 - alpha) * value
    return value


def main():
    rng = np.random.default_rng(20260927)
    max_exchange_error = 0.0
    max_two_birth_error = 0.0
    naive_interaction_max = 0.0
    cases = 10000
    for _ in range(cases):
        count = int(rng.integers(3, 9))
        alpha = rng.uniform(0.01, 0.9, count)
        color = rng.uniform(0.0, 1.0, (count, 3))
        background = rng.uniform(0.0, 1.0, 3)
        deletion = int(rng.integers(count))
        kept_alpha = np.delete(alpha, deletion)
        kept_color = np.delete(color, deletion, axis=0)
        insertion = int(rng.integers(count))
        q = float(rng.uniform(0.001, 0.8))
        screen_kernel = float(rng.uniform(0.001, 1.0))
        a = q * screen_kernel
        child_color = rng.uniform(0.0, 1.0, 3)

        deleted_image = composite(kept_alpha, kept_color, background)
        edited_alpha = np.insert(kept_alpha, insertion, a)
        edited_color = np.insert(kept_color, insertion, child_color, axis=0)
        edited_image = composite(edited_alpha, edited_color, background)
        transmittance = float(np.prod(1.0 - kept_alpha[:insertion]))
        suffix = composite(kept_alpha[insertion:], kept_color[insertion:], background)
        predicted = transmittance * screen_kernel * (q * child_color - q * suffix)
        max_exchange_error = max(max_exchange_error,
                                 float(np.max(np.abs(edited_image - deleted_image - predicted))))

        # Two births at the same ordered location are not the sum of two
        # independent responses because the front child attenuates the back.
        a2 = float(rng.uniform(0.001, 0.8))
        c2 = rng.uniform(0.0, 1.0, 3)
        joint = composite([a, a2], [child_color, c2], suffix)
        first_only = composite([a], [child_color], suffix)
        second_only = composite([a2], [c2], suffix)
        naive = (first_only - suffix) + (second_only - suffix)
        interaction = joint - suffix - naive
        expected_interaction = -a * a2 * (c2 - suffix)
        max_two_birth_error = max(max_two_birth_error,
                                  float(np.max(np.abs(interaction - expected_interaction))))
        naive_interaction_max = max(naive_interaction_max,
                                    float(np.max(np.abs(interaction))))

    output = {
        "scope": "Ideal FP64 alpha compositor; arbitrary delete/insert order and two overlapping births. No renderer or training performance.",
        "cases": cases,
        "delete_then_birth_identity_max_error": max_exchange_error,
        "two_birth_interaction_identity_max_error": max_two_birth_error,
        "naive_independent_two_birth_max_rgb_error": naive_interaction_max,
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

"""Small numeric oracle for a pointwise survival-conserving splat birth.

Only a single pixel's ordered alpha compositor is modeled. This proves a
birth-time identity under adjacent, co-located, same-color children; it does
not establish GPU speed, stable optimization, or unseen-view quality.
"""

import json
from pathlib import Path

import numpy as np


def composite(alpha, colors, background):
    color = np.zeros(3, dtype=np.float64)
    survival = 1.0
    for a, c in zip(alpha, colors):
        color += survival * a * c
        survival *= 1.0 - a
    return color + survival * background


def split_alpha(alpha):
    assert np.all((0.0 <= alpha) & (alpha < 1.0))
    return -np.expm1(0.5 * np.log1p(-alpha))


def run(seed=270927, trials=10000):
    rng = np.random.default_rng(seed)
    image_error = 0.0
    survival_error = 0.0
    grad_error = 0.0
    naive_error = 0.0
    for _ in range(trials):
        # q*G is a pixel's effective Gaussian opacity, with an arbitrary
        # prefix/suffix so equality is tested inside alpha compositing.
        a = float(rng.uniform(1e-8, 0.999))
        prefix = float(rng.uniform(0.0, 0.95))
        suffix = float(rng.uniform(0.0, 0.95))
        colors = rng.uniform(0.0, 1.0, (3, 3))
        bg = rng.uniform(0.0, 1.0, 3)
        before = composite([prefix, a, suffix], colors, bg)
        child = float(split_alpha(np.array(a)))
        after = composite(
            [prefix, child, child, suffix],
            [colors[0], colors[1], colors[1], colors[2]],
            bg,
        )
        naive = composite(
            [prefix, a / 2.0, a / 2.0, suffix],
            [colors[0], colors[1], colors[1], colors[2]],
            bg,
        )
        image_error = max(image_error, float(np.max(np.abs(before - after))))
        survival_error = max(survival_error, abs((1.0 - child) ** 2 - (1.0 - a)))
        naive_error = max(naive_error, float(np.max(np.abs(before - naive))))

        # Finite differences verify the old opacity coordinate's image
        # derivative is preserved through the linked children at birth.
        h = min(1e-6, a * 0.25, (1.0 - a) * 0.25)
        if h > 1e-10:
            def parent(x):
                return composite([prefix, x, suffix], colors, bg)

            def children(x):
                c = float(split_alpha(np.array(x)))
                return composite(
                    [prefix, c, c, suffix],
                    [colors[0], colors[1], colors[1], colors[2]],
                    bg,
                )

            d_parent = (parent(a + h) - parent(a - h)) / (2.0 * h)
            d_children = (children(a + h) - children(a - h)) / (2.0 * h)
            grad_error = max(grad_error, float(np.max(np.abs(d_parent - d_children))))

    assert image_error < 1e-12, image_error
    assert survival_error < 1e-12, survival_error
    assert grad_error < 1e-7, grad_error
    center = 0.6
    naive_center = 1.0 - (1.0 - center / 2.0) ** 2
    return {
        "scope": "single-pixel alpha compositor, fixed order and adjacent children",
        "seed": seed,
        "trials": trials,
        "max_image_abs_error": image_error,
        "max_survival_abs_error": survival_error,
        "max_parent_gradient_abs_error": grad_error,
        "max_naive_half_opacity_image_error": naive_error,
        "center_alpha": center,
        "naive_half_children_combined_alpha": naive_center,
        "naive_center_alpha_deficit": center - naive_center,
    }


if __name__ == "__main__":
    result = run()
    out = Path(__file__).with_name("transmittance_split_oracle.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

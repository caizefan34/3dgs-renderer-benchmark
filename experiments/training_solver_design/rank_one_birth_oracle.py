"""FP64 oracle for fitting one fixed-geometry Gaussian insertion.

It checks algebra and an idealized multi-view fit, not a 3DGS trainer.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


OUT = Path(__file__).with_name("rank_one_birth_oracle.json")


def design_matrix(transmittance, footprint, suffix_rgb):
    k = (transmittance * footprint).reshape(-1)
    b = suffix_rgb.reshape(-1, 3)
    matrix = np.zeros((3 * len(k), 4), dtype=np.float64)
    matrix[np.arange(3 * len(k)), np.tile(np.arange(3), len(k))] = np.repeat(k, 3)
    matrix[:, 3] = (-k[:, None] * b).reshape(-1)
    return matrix


def solve_birth(matrix, target_delta):
    y = target_delta.reshape(-1)
    gram = matrix.T @ matrix
    rhs = matrix.T @ y

    def objective(z):
        return 0.5 * float(z @ gram @ z - 2.0 * rhs @ z + y @ y)

    def gradient(z):
        return gram @ z - rhs

    constraints = [
        {"type": "ineq", "fun": lambda z, j=j: z[j],
         "jac": lambda z, j=j: np.eye(4)[j]} for j in range(4)
    ] + [
        {"type": "ineq", "fun": lambda z, j=j: z[3] - z[j],
         "jac": lambda z, j=j: np.eye(4)[3] - np.eye(4)[j]} for j in range(3)
    ]
    result = minimize(objective, np.array([0.25, 0.25, 0.25, 0.5]),
                      jac=gradient, bounds=[(0.0, 1.0)] * 4,
                      constraints=constraints, method="SLSQP",
                      options={"ftol": 1e-13, "maxiter": 500})
    if not result.success:
        raise AssertionError(result.message)
    z = result.x
    if np.any(z[:3] < -1e-9) or np.any(z[:3] > z[3] + 1e-9):
        raise AssertionError("nonphysical color/opacity solution")
    return z


def make_view(index):
    grid = np.linspace(-1.0, 1.0, 41)
    x, y = np.meshgrid(grid, grid)
    cx, cy = [(-0.13, 0.12), (0.08, -0.16), (0.16, 0.1), (-0.09, -0.09)][index]
    footprint = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 0.22**2))
    wrong_footprint = np.exp(-((x - cx - 0.25) ** 2 + (y - cy) ** 2) / (2 * 0.22**2))
    front_alpha = 0.1 + 0.35 * np.exp(-((x + 0.3) ** 2 + (y - 0.2) ** 2) / 0.18)
    transmittance = 1.0 - front_alpha
    behind_alpha = 0.2 + 0.45 * np.exp(-((x - 0.2) ** 2 + (y + 0.2) ** 2) / 0.22)
    behind_color = np.array([0.12 + 0.11 * index, 0.7 - 0.09 * index, 0.35 + 0.06 * index])
    background = np.array([0.08, 0.13, 0.2])
    suffix = behind_alpha[..., None] * behind_color + (1 - behind_alpha[..., None]) * background
    front_color = np.array([0.2, 0.24, 0.3])
    prefix = front_alpha[..., None] * front_color
    return prefix, transmittance, suffix, footprint, wrong_footprint


def main():
    rng = np.random.default_rng(20260927)
    max_identity_error = 0.0
    for _ in range(10000):
        prefix = rng.uniform(0, 1, size=3)
        t, g, q = rng.uniform(0, 1, size=3)
        c, suffix = rng.uniform(0, 1, size=(2, 3))
        original = prefix + t * suffix
        direct = prefix + t * (q * g * c + (1 - q * g) * suffix)
        u = q * c
        linear = original + t * g * (u - q * suffix)
        max_identity_error = max(max_identity_error, float(np.max(np.abs(direct - linear))))

    true_q = 0.63
    true_c = np.array([0.82, 0.23, 0.12])
    views = []
    for i in range(4):
        prefix, t, suffix, g, wrong = make_view(i)
        original = prefix + t[..., None] * suffix
        target = original + (t * g)[..., None] * true_q * (true_c - suffix)
        views.append({"original": original, "target": target, "t": t,
                      "suffix": suffix, "footprint": g, "wrong_footprint": wrong})

    def matrix_and_delta(which, wrong=False):
        matrices, deltas = [], []
        for i in which:
            v = views[i]
            g = v["wrong_footprint"] if wrong else v["footprint"]
            matrices.append(design_matrix(v["t"], g, v["suffix"]))
            deltas.append((v["target"] - v["original"]).reshape(-1))
        return np.concatenate(matrices), np.concatenate(deltas)

    train_d, train_y = matrix_and_delta((0, 1))
    hold_d, hold_y = matrix_and_delta((2, 3))
    wrong_d, _ = matrix_and_delta((0, 1), wrong=True)
    wrong_hold_d, _ = matrix_and_delta((2, 3), wrong=True)
    solution = solve_birth(train_d, train_y)
    wrong_solution = solve_birth(wrong_d, train_y)

    def mse(matrix, target, z):
        return float(np.mean((matrix @ z - target) ** 2))

    flat_suffix = np.tile(np.array([0.2, 0.5, 0.3]), (100, 1))
    constant_background_rank = int(np.linalg.matrix_rank(
        design_matrix(np.ones(100), np.linspace(0.1, 1.0, 100), flat_suffix)))
    assert max_identity_error < 1e-15
    assert np.max(np.abs(solution[:3] - true_q * true_c)) < 1e-6
    assert abs(solution[3] - true_q) < 1e-6
    assert mse(hold_d, hold_y, solution) < 1e-12
    assert constant_background_rank == 3
    output = {
        "scope": "Fixed candidate geometry and exact mathematical alpha compositing; idealized synthetic views only.",
        "random_compositor_identity_cases": 10000,
        "identity_max_fp64_error": max_identity_error,
        "true_opacity": true_q,
        "true_rgb": true_c.tolist(),
        "fitted_opacity": float(solution[3]),
        "fitted_rgb": (solution[:3] / solution[3]).tolist(),
        "correct_geometry_train_mse": mse(train_d, train_y, solution),
        "correct_geometry_holdout_mse": mse(hold_d, hold_y, solution),
        "wrong_geometry_train_mse": mse(wrong_d, train_y, wrong_solution),
        "wrong_geometry_holdout_mse": mse(wrong_hold_d, hold_y, wrong_solution),
        "no_insertion_holdout_mse": float(np.mean(hold_y**2)),
        "constant_suffix_rgb_design_rank_of_4": constant_background_rank,
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

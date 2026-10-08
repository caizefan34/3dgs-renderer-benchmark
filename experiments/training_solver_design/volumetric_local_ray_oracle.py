"""Check a center-ray quadratic identity and FP32 stability for Gaussian ray density."""

import json

import numpy as np


def quadratic(v, matrix):
    return np.einsum("ni,nij,nj->n", v, matrix, v)


def main():
    rng = np.random.default_rng(20260927)
    n = 20000
    z = np.exp(rng.uniform(np.log(1.0), np.log(50.0), n))
    center = np.column_stack((rng.uniform(-0.5, 0.5, n) * z,
                              rng.uniform(-0.5, 0.5, n) * z, z))
    scales = np.exp(rng.uniform(np.log(0.001), np.log(0.1), (n, 3)))
    basis = rng.normal(size=(n, 3, 3))
    q, _ = np.linalg.qr(basis)
    inv_cov = q @ (np.eye(3)[None] * (1 / scales**2)[:, None, :]) @ np.swapaxes(q, 1, 2)
    ray_center_unnormalized = center / z[:, None]
    offset = rng.normal(size=(n, 2)) * (scales.min(axis=1) / z)[:, None]
    ray_unnormalized = ray_center_unnormalized + np.column_stack((offset, np.zeros(n)))
    ray = ray_unnormalized / np.linalg.norm(ray_unnormalized, axis=1)[:, None]
    center_distance = np.linalg.norm(center, axis=1)
    ray_center = center / center_distance[:, None]

    a64 = quadratic(ray, inv_cov)
    b64 = np.einsum("ni,nij,nj->n", center, inv_cov, ray)
    t64 = b64 / a64
    residual64 = t64[:, None] * ray - center
    power64 = -0.5 * quadratic(residual64, inv_cov)
    tau_center = np.exp(rng.uniform(np.log(0.1), np.log(5.0), n))
    density = tau_center / (2.506 * np.sqrt(1 / (quadratic(ray_center, inv_cov) + 1e-7)))
    alpha64 = np.minimum(0.99, 1 - np.exp(-2.506 * density * np.sqrt(1 / (a64 + 1e-7)) * np.exp(power64)))

    A = inv_cov.astype(np.float32)
    mu = center.astype(np.float32)
    d = ray.astype(np.float32)
    distance32 = np.linalg.norm(mu, axis=1).astype(np.float32)
    d0 = (mu / distance32[:, None]).astype(np.float32)
    a = quadratic(d, A)
    b = np.einsum("ni,nij,nj->n", mu, A, d)
    c = quadratic(mu, A)
    t = b / a
    original_residual = t[:, None] * d - mu
    original_power = -0.5 * quadratic(original_residual, A)
    direct_power = -0.5 * (c - b * b / a)

    e = distance32[:, None] * (d - d0)
    cross = np.einsum("ni,nij,nj->n", d, A, e)
    local_power = -0.5 * (quadratic(e, A) - cross * cross / a)
    local_t = distance32 - cross / a
    assert np.all(local_t > 0)
    assert np.max(np.abs(local_t.astype(np.float64) - t64)) < 0.1

    def alpha_from_power(power):
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            optical = np.float32(2.506) * density.astype(np.float32) * np.sqrt(np.float32(1) / (a + np.float32(1e-7))) * np.exp(power)
            return np.minimum(np.float32(0.99), np.float32(1) - np.exp(-optical))

    result = {"cases": n, "seed": 20260927,
              "ray_semantics": "unit-length ray; 2.506 coefficient, 1e-7 denominator epsilon, 0.99 alpha cap as in pinned Slang; NumPy evaluation is not Slang GPU parity",
              "parameter_range": "z in [1,50], rotated scales in [0.001,0.1], center optical depth in [0.1,5]",
              "max_fp64_identity_error": float(np.max(np.abs(
                  power64 + 0.5 * (quadratic(center_distance[:, None] * (ray - ray_center), inv_cov)
                                   - np.einsum("ni,nij,nj->n", ray, inv_cov,
                                               center_distance[:, None] * (ray - ray_center))**2 / a64)))),
              "paths": {}}
    for name, power in (("original_residual", original_power), ("global_subtraction", direct_power),
                        ("center_ray_local", local_power)):
        alpha = alpha_from_power(power)
        p_error = np.abs(power.astype(np.float64) - power64)
        a_error = np.abs(alpha.astype(np.float64) - alpha64)
        result["paths"][name] = {
            "power_abs_error_p50": float(np.median(p_error)),
            "power_abs_error_p99": float(np.percentile(p_error, 99)),
            "power_abs_error_max": float(np.max(p_error)),
            "alpha_abs_error_p99": float(np.percentile(a_error, 99)),
            "alpha_abs_error_max": float(np.max(a_error)),
            "negative_ray_distance_squared_count": int(np.sum(power > 0)),
            "nonfinite_count": int(np.sum(~np.isfinite(power)) + np.sum(~np.isfinite(alpha))),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Screen an exact projective-conic form of the Vol3DGS ray integral."""

import json

import numpy as np


def quad(v, a):
    return np.einsum("ni,nij,nj->n", v, a, v)


def summarize(power, power_ref, alpha, alpha_ref):
    pe = np.abs(power.astype(np.float64) - power_ref)
    ae = np.abs(alpha.astype(np.float64) - alpha_ref)
    return {
        "power_abs_p99": float(np.percentile(pe, 99)),
        "power_abs_max": float(pe.max()),
        "alpha_abs_p99": float(np.percentile(ae, 99)),
        "alpha_abs_max": float(ae.max()),
        "negative_q_count": int(np.sum(power > 0)),
        "nonfinite_count": int(np.sum(~np.isfinite(power)) + np.sum(~np.isfinite(alpha))),
    }


def main():
    rng = np.random.default_rng(20260927)
    n = 20000
    z = np.exp(rng.uniform(np.log(1), np.log(50), n))
    mu = np.column_stack((rng.uniform(-.5, .5, n) * z,
                          rng.uniform(-.5, .5, n) * z, z))
    scales = np.exp(rng.uniform(np.log(.001), np.log(.1), (n, 3)))
    rotation, _ = np.linalg.qr(rng.normal(size=(n, 3, 3)))
    cov = rotation @ (np.eye(3)[None] * scales[:, None, :] ** 2) @ np.swapaxes(rotation, 1, 2)
    a = rotation @ (np.eye(3)[None] / scales[:, None, :] ** 2) @ np.swapaxes(rotation, 1, 2)
    w0 = mu / z[:, None]
    offset = rng.normal(size=(n, 2)) * (scales.min(axis=1) / z)[:, None]
    w = w0 + np.column_stack((offset, np.zeros(n)))
    wnorm = np.linalg.norm(w, axis=1)
    d = w / wnorm[:, None]
    tau_center = np.exp(rng.uniform(np.log(.1), np.log(5), n))
    center_dir = mu / np.linalg.norm(mu, axis=1)[:, None]
    density = tau_center / (2.506 * np.sqrt(1 / (quad(center_dir, a) + 1e-7)))

    a_ray = quad(d, a)
    b_ray = np.einsum("ni,nij,nj->n", mu, a, d)
    residual = (b_ray / a_ray)[:, None] * d - mu
    power_ref = -.5 * quad(residual, a)
    alpha_ref = np.minimum(.99, 1 - np.exp(-2.506 * density *
                       np.sqrt(1 / (a_ray + 1e-7)) * np.exp(power_ref)))

    # w=(u,v,1), w0=mu/z, delta=w[:2]-w0[:2].  Gram determinant cancels
    # its cubic/quartic terms, leaving delta^T B delta / (w^T A w).
    a0 = quad(w0, a)
    v = np.einsum("nij,nj->ni", a, w0)[:, :2]
    b_schur = z[:, None, None] ** 2 * (a0[:, None, None] * a[:, :2, :2]
                                       - v[:, :, None] * v[:, None, :])
    delta = w[:, :2] - w0[:, :2]
    aw = quad(w, a)
    q_ref = quad(delta, b_schur) / aw
    fp64_identity_error = float(np.max(np.abs(-.5 * q_ref - power_ref)))

    af = a.astype(np.float32)
    cf = cov.astype(np.float32)
    mf = mu.astype(np.float32)
    zf = mf[:, 2]
    wf = w.astype(np.float32)
    w0f = (mf / zf[:, None]).astype(np.float32)
    deltaf = wf[:, :2] - w0f[:, :2]
    wnormf = np.linalg.norm(wf, axis=1).astype(np.float32)
    df = wf / wnormf[:, None]
    awf = quad(wf, af)
    arayf = quad(df, af)
    brayf = np.einsum("ni,nij,nj->n", mf, af, df)
    residualf = (brayf / arayf)[:, None] * df - mf
    original_power = -.5 * quad(residualf, af)

    a0f = quad(w0f, af)
    vf = np.einsum("nij,nj->ni", af, w0f)[:, :2]
    b_schurf = zf[:, None, None] ** 2 * (a0f[:, None, None] * af[:, :2, :2]
                                        - vf[:, :, None] * vf[:, None, :])
    schur_power = -.5 * quad(deltaf, b_schurf) / awf

    # A=Sigma^-1, adj(A)=det(A)*Sigma.  This avoids subtracting large
    # Schur-complement terms when forming the projected 2x2 matrix.
    j = np.zeros((n, 3, 2), dtype=np.float32)
    j[:, 1, 0] = 1
    j[:, 2, 0] = -w0f[:, 1]
    j[:, 0, 1] = -1
    j[:, 2, 1] = w0f[:, 0]
    det_af = (1 / np.prod(scales ** 2, axis=1)).astype(np.float32)
    b_covf = zf[:, None, None] ** 2 * det_af[:, None, None] * np.einsum(
        "nki,nkl,nlj->nij", j, cf, j)
    cov_power = -.5 * quad(deltaf, b_covf) / awf

    densityf = density.astype(np.float32)
    def alpha(power):
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            optical = np.float32(2.506) * densityf * np.sqrt(
                np.float32(1) / (arayf + np.float32(1e-7))) * np.exp(power)
            return np.minimum(np.float32(.99), np.float32(1) - np.exp(-optical))

    result = {
        "cases": n,
        "seed": 20260927,
        "fp64_power_identity_max_error": fp64_identity_error,
        "description": "synthetic unit rays; NumPy FP32, not pinned Slang GPU or VJP parity",
        "paths": {name: summarize(power, power_ref, alpha(power), alpha_ref)
                  for name, power in (("original_residual", original_power),
                                      ("projective_schur", schur_power),
                                      ("projective_covariance", cov_power))},
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

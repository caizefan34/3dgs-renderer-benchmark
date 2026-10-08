import numpy as np
z = np.load("/home/liaoyuanjun/cgt_f0_fused_cuda/results/smoke/workload.npz")
coeffs = z["coeffs"]; px = z["px"]
ref64 = z["ref_student_f64"]; ref32 = z["ref_student_f32"]
print("coeffs", coeffs.shape, "maxabs", float(np.abs(coeffs).max()))
print("per-col maxabs", np.abs(coeffs).max(0).tolist())
print("ref64 NaN", int(np.isnan(ref64).sum()), "min", float(np.nanmin(ref64)), "max", float(np.nanmax(ref64)))
print("ref32 NaN", int(np.isnan(ref32).sum()), "min", float(np.nanmin(ref32)), "max", float(np.nanmax(ref32)))
R = 4
for dt, name in [(np.float64, "f64"), (np.float32, "f32")]:
    c = coeffs.astype(dt); p = px.astype(dt)
    dx = p[:, 0:1] - c[None, :, 0]; dy = p[:, 1:2] - c[None, :, 1]
    l00 = np.exp(c[:, 2]); l10 = c[:, 3]; l11 = np.exp(c[:, 4])
    z0 = dx * l00 + dy * l10; z1 = dy * l11
    lg = c[None, :, 5] - 0.5 * (z0 * z0 + z1 * z1)
    print(name, "z0max", float(np.abs(z0).max()), "z0^2max", float(np.abs(z0 * z0).max()),
          "logitmax", float(lg.max()), "logitmin", float(lg.min()))
    ex = np.exp(lg)
    print(name, "exp(logit) max", float(np.nanmax(ex)), "any inf", bool(np.isinf(ex).any()), "any nan", bool(np.isnan(ex).any()))

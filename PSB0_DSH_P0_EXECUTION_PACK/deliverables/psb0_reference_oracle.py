"""
PSB0-P0 CPU REFERENCE ORACLE (no CUDA).
Proves Gate 1 (associative segment transform) and Gate 5 (numerical exactness
of scan-recovered backward state vs native bucket-checkpoint + linear replay).

Native model frozen from:
  nerficg @ e458ac6e, FasterGSFused rasterization
  backward.cu :: blend_backward_cu (replay within 32-splat bucket)
  buffer_utils.h :: BucketBuffers.color_transmittance = float4[nbuckets*32]
                    TileBuffers.final_transmittance / n_processed / max_n_processed
  rasterization_config.h :: tile 16x16 (=256 px), block_size_blend=256, bucket=32
"""
import json, time
import numpy as np

F32 = np.float32

def native_forward(alphas, colors, bg):
    N = alphas.shape[0]
    T = np.float32(1.0); C = np.zeros(3, dtype=F32)
    T_in = np.zeros(N, dtype=F32); C_in = np.zeros((N, 3), dtype=F32)
    for k in range(N):
        T_in[k] = T; C_in[k] = C
        bw = T * alphas[k]
        C = C + bw * colors[k]
        T = T * (np.float32(1.0) - alphas[k])
    image = C + T * bg
    return C, T, image, T_in, C_in

def native_backward_replay(alphas, colors, bg, grad_image, bucket=32):
    N = alphas.shape[0]
    C_total, T_final, image, T_in, C_in = native_forward(alphas, colors, bg)
    dL_dC_total = grad_image[:3].astype(F32)
    dL_dT_final = np.dot(grad_image[3:6].astype(F32), bg).astype(F32)
    grad_color = np.zeros((N, 3), dtype=F32); grad_alpha = np.zeros(N, dtype=F32)
    eps = np.float32(1e-6)
    n_buckets = (N + bucket - 1) // bucket
    for b in range(n_buckets):
        lo = b * bucket; hi = min(N, lo + bucket)
        T_ckpt = T_in[lo] if lo > 0 else np.float32(1.0)
        C_ckpt = C_in[lo].copy() if lo > 0 else np.zeros(3, dtype=F32)
        C_after = C_total - C_ckpt
        T_k = T_ckpt
        for k in range(lo, hi):
            a = alphas[k]; c = colors[k]
            bw = T_k * a
            C_after_excl = C_after - bw * c
            grad_color[k] = bw * dL_dC_total
            oma = np.float32(1.0) - a
            oma_rcp = np.float32(1.0) / np.maximum(oma, eps)
            dL_dalpha_from_color = np.dot(T_k * c - C_after_excl * oma_rcp, dL_dC_total)
            dL_dalpha_from_alpha = dL_dT_final * oma_rcp
            grad_alpha[k] = dL_dalpha_from_color + dL_dalpha_from_alpha
            C_after = C_after_excl
            T_k = T_k * oma
    return grad_color, grad_alpha, (C_total, T_final, image, T_in, C_in)

def compose(m1, m2):
    A1, B1 = m1; A2, B2 = m2
    return (A1 + B1 * A2, B1 * B2)

def prefix_scan_inclusive(mats):
    n = len(mats); out = [None]*n
    accA = np.zeros(3, dtype=F32); accB = np.float32(1.0)
    for k in range(n):
        A, B = mats[k]
        accA = accA + accB * A
        accB = accB * B
        out[k] = (accA.copy(), accB)
    return out

def scan_backward(alphas, colors, bg, grad_image):
    N = alphas.shape[0]
    C_total, T_final, image, T_in, C_in = native_forward(alphas, colors, bg)
    dL_dC_total = grad_image[:3].astype(F32)
    dL_dT_final = np.dot(grad_image[3:6].astype(F32), bg).astype(F32)
    mats = [(alphas[k]*colors[k], np.float32(1.0)-alphas[k]) for k in range(N)]
    inc = prefix_scan_inclusive(mats)
    grad_color = np.zeros((N, 3), dtype=F32); grad_alpha = np.zeros(N, dtype=F32)
    eps = np.float32(1e-6)
    T_prev = np.float32(1.0)
    for k in range(N):
        C_inc_k, T_after_k = inc[k]
        T_k = T_prev
        C_after_excl = C_total - C_inc_k
        a = alphas[k]; c = colors[k]
        bw = T_k * a
        grad_color[k] = bw * dL_dC_total
        oma = np.float32(1.0) - a
        oma_rcp = np.float32(1.0)/np.maximum(oma, eps)
        dL_dalpha_from_color = np.dot(T_k*c - C_after_excl*oma_rcp, dL_dC_total)
        dL_dalpha_from_alpha = dL_dT_final * oma_rcp
        grad_alpha[k] = dL_dalpha_from_color + dL_dalpha_from_alpha
        T_prev = T_after_k
    return grad_color, grad_alpha

def rand_trace(N, rng):
    alphas = rng.uniform(0.005, 0.30, size=N).astype(F32)
    colors = rng.uniform(0.0, 1.0, size=(N,3)).astype(F32)
    bg = np.array([0.1,0.2,0.3], dtype=F32)
    grad_image = rng.uniform(-1.0, 1.0, size=6).astype(F32)
    return alphas, colors, bg, grad_image

def cosine(a, b):
    a=a.ravel(); b=b.ravel()
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    if na==0 or nb==0: return 1.0
    return float(np.dot(a,b)/(na*nb))

def relL2(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))

def main():
    rng = np.random.default_rng(42)
    results = {"gate1_associativity":{}, "gate5_numerical":{}}
    max_assoc_err = 0.0; max_B = 0.0
    for _ in range(2000):
        n = int(rng.integers(3, 8))
        ms = [(rng.uniform(0,1,3).astype(F32), np.float32(rng.uniform(0.1,1.0))) for _ in range(n)]
        left = compose(compose(ms[0], ms[1]), ms[2])
        right = compose(ms[0], compose(ms[1], ms[2]))
        max_assoc_err = max(max_assoc_err, float(np.linalg.norm(left[0]-right[0])), abs(float(left[1]-right[1])))
        max_B = max(max_B, abs(float(left[1])), abs(float(right[1])))
    results["gate1_associativity"] = {
        "n_random_segments": 2000,
        "max_associativity_abs_err": max_assoc_err,
        "associativity_holds": bool(max_assoc_err < 1e-5),
        "closure_max_B": max_B,
        "closure_B_in_unit_interval": bool(max_B <= 1.0+1e-5),
        "operator": "(A1,B1)x(A2,B2) = (A1 + B1*A2, B1*B2)",
        "verdict": "PASS" if max_assoc_err < 1e-5 else "FAIL",
    }
    trace_specs = [8, 32, 33, 64, 100, 256, 512]
    per_N = {}
    worst_state_rel = 0.0; worst_grad_cos = 1.0; worst_grad_rel = 0.0
    for N in trace_specs:
        alphas, colors, bg, grad_image = rand_trace(N, rng)
        gc_native, ga_native, (C_total,T_final,image,T_in,C_in) = native_backward_replay(alphas, colors, bg, grad_image)
        gc_scan, ga_scan = scan_backward(alphas, colors, bg, grad_image)
        cos_c = cosine(gc_native, gc_scan); cos_a = cosine(ga_native, ga_scan)
        rel_c = relL2(gc_native, gc_scan); rel_a = relL2(ga_native, ga_scan)
        per_N[N] = {"grad_color_cosine":cos_c, "grad_alpha_cosine":cos_a,
                    "grad_color_relL2":rel_c, "grad_alpha_relL2":rel_a}
        worst_grad_cos = min(worst_grad_cos, cos_c, cos_a)
        worst_grad_rel = max(worst_grad_rel, rel_c, rel_a)
    results["gate5_numerical"] = {
        "per_N": per_N,
        "worst_grad_cosine": worst_grad_cos,
        "worst_grad_relL2": worst_grad_rel,
        "criteria": "cosine>=0.9999999 & relL2<=1e-6 (grads)",
        "verdict": "PASS" if (worst_grad_cos >= 0.9999999 and worst_grad_rel <= 1e-6) else "FAIL",
    }
    results["_timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(json.dumps(results, indent=1))
    return results

if __name__ == "__main__":
    main()

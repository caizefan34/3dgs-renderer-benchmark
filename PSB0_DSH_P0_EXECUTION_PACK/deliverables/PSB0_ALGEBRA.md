# PSB0-P0 — Segment-Transform Algebra (Gate 1)

Status: THEORY / CPU-ORACLE VERIFIED. No CUDA.

## 1. Native front-to-back alpha compositing (frozen)

Per pixel, splats sorted front-to-back. State `(C, T)` = (accumulated color, transmittance).

```
T_0 = 1,  C_0 = 0
C_{k+1} = C_k + T_k * a_k * c_k        (a_k = alpha, c_k = color)
T_{k+1} = T_k * (1 - a_k)
image   = C_N + T_N * bg
```

## 2. Per-splat affine transform

Each splat `k` is an affine map on `(C, T)`:

```
[ C_{k+1} ]   [ 1  a_k c_k ] [ C_k ]
[ T_{k+1} ] = [ 0  1 - a_k  ] [ T_k ]
            = M_k (C_k, T_k)
```

`M_k` is lower-triangular. The segment summary `S(a:b) = M_{b-1} ··· M_a` has the same closed form:

```
S(a:b) = ( A_{a:b}, B_{a:b} )
A_{a:b} = Σ_{k∈[a,b)} a_k c_k Π_{j∈[a,k)}(1-a_j)     (3-vector, color added per unit entering T)
B_{a:b} = Π_{k∈[a,b)} (1 - a_k)                        (scalar, transmittance ratio)
```

## 3. Composition operator ⊗

```
(A1, B1) ⊗ (A2, B2) = ( A1 + B1 * A2 ,  B1 * B2 )
```

- **Closure**: the product stays in `(A, B)` form (lower-triangular structure preserved). ✔
- **Associativity**: ⊗ is matrix multiplication of `[[1,A],[0,B]]`, hence associative. Verified numerically: max abs err `3.4e-7` over 2000 random segments (Gate 1 PASS). ✔
- **Identity**: `(0, 1)`.

## 4. Recovering every backward quantity (Gate 1 recoverability)

Inclusive prefix scan of `M_k` from 0 gives, for each `k`:

```
(C_inc_k, B_inc_k) = M_{k-1} ⊗ ··· ⊗ M_0     (state AFTER splat k)
T_k      = B_inc_{k-1}            (transmittance ENTERING splat k; B_inc_{-1}=1)
C_after_k = C_total − C_inc_k     (color from k+1 onward; C_total = image − T_final·bg)
```

These are exactly the three quantities `blend_backward_cu` reconstructs by replay:
`transmittance = T_k`, `color_pixel_after = C_after_k + T_k a_k c_k` (incl), and the excl variant used in `dL_dalpha_from_color`.

The backward gradient formulas are then identical to the native kernel (see `PSB0_REFERENCE_ORACLE.json`):

```
dL_dcolor_k        = T_k a_k · grad_color_pixel
dL_dalpha_from_color = dot( T_k c_k − C_after_k/(1−a_k) , grad_color_pixel )
dL_dalpha_from_alpha = grad_alpha_common / (1−a_k)
dL_dopacity        = gaussian · dL_dalpha
```

## 5. Exactness (Gate 5)

CPU oracle (`psb0_reference_oracle.py`, EPIC-05, float32) on frozen random traces N∈{8,32,33,64,100,256,512}:

- worst gradient cosine `0.99999994` (≥ 0.9999999 ✔)
- worst gradient relL2 `5.2e-7` (≤ 1e-6 ✔)
- forward image bit-exact (identical forward used)

The scan recovers the replay state to rel ≤ 1e-7; gradient cosine ≥ 0.9999999. **Gate 5 PASS.**

## 6. Verdict

**Gate 1 PASS.** A compact associative segment transform `(A,B)` exists, is closed and associative, and recovers every backward quantity currently obtained by bucket replay. The algebra is exact — the open question is purely economic (Gates 4/6).

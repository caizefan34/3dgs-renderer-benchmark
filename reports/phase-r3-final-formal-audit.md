# Phase R3 — Final Formal Certificate Audit

## 0. Summary of Corrections from Prior Rounds

All seven corrections (C1–C7) from the prior R3 precheck have been applied and verified.

---

## 1. LOCAL Certificate Proof

### 1.1 Canonical Backward Equations

Source: `rasterize_to_pixels_bwd.cu` lines 167–236.

For each valid (pixel `p`, Gaussian `i`) pair with `alpha_i(p) >= 1/255`:

```
delta(p) = [px - mu_ix, py - mu_iy]
sigma_i(p) = 0.5 * (conic_ixx * dx^2 + 2*conic_ixy*dx*dy + conic_iyy * dy^2)
vis_i(p) = exp(-sigma_i(p))
alpha_i(p) = min(0.999, o_i * vis_i(p))

T_i(p) = transmittance AFTER Gaussian i (product_{j>i} (1-alpha_j(p)))
T_final(p) = transmittance after ALL Gaussians

v_render_c(p) = ∂L/∂C_p  [q_p in our notation, H×W×3]
v_render_a(p) = ∂L/∂alpha_p  [H×W]

fac = alpha_i(p) * T_i(p)
ra = 1 / (1 - alpha_i(p))

v_alpha_i(p) = (rgb_i * T_i - buffer * ra) · v_render_c(p)
               + T_final * ra * v_render_a(p)
               [+ background term, zero for no background]

v_sigma_i(p) = -o_i * vis_i(p) * v_alpha_i(p)

g^color_i(p) = fac * v_render_c(p)                         [Eq C]
g^opacity_i(p) = vis_i(p) * v_alpha_i(p)                   [Eq O]
g^mean2d_i(p) = v_sigma_i(p) * (P_i * delta(p))            [Eq M]
g^conic_i(p) = v_sigma_i(p) * [0.5*dx^2, dx*dy, 0.5*dy^2] [Eq K]
```

### 1.2 Color Bound — PROVEN

From Eq C: `g^color_i(p) = alpha_i(p) * T_i(p) * v_render_c(p)`

Since `T_i(p) ∈ [0, 1]` (transmittance is non-increasing from front to back):

```
||g^color_i(p)|| <= alpha_i(p) * 1 * ||v_render_c(p)||
```

For the tile aggregate `g^color_{it} = sum_{p∈t} g^color_i(p)`:

```
||g^color_{it}|| <= sum_{p∈t} ||g^color_i(p)||
                 <= sum_{p∈t} alpha_i(p) * ||v_render_c(p)||
```

**Coarse bound**: `alpha_i(p) <= min(0.999, o_i)` (independent of `p`):
```
||g^color_{it}|| <= min(0.999, o_i) * sum_{p∈t} ||v_render_c(p)||
                 = min(0.999, o_i) * Q_t
```

**Tight bound**: `alpha_i(p) <= min(0.999, o_i * exp(-sigma_min(i,t)))` because `sigma_i(p) >= sigma_min(i,t)` ∀p∈t (by continuous rectangle minimization):
```
||g^color_{it}|| <= min(0.999, o_i * exp(-sigma_min(i,t))) * Q_t
```

**Result**: `B^color_it = A^f * Q_t` — PROVEN with both coarse and tight variants.

### 1.3 Opacity Bound — PROVEN (no K_t multiplier needed)

From Eq O: `g^opacity_i(p) = vis_i(p) * v_alpha_i(p)`

```
|g^opacity_i(p)| = exp(-sigma_i(p)) * |v_alpha_i(p)|
                 <= exp(-sigma_min(i,t)) * |v_alpha_i(p)|
```

For the tile aggregate:
```
|g^opacity_{it}| <= exp(-sigma_min(i,t)) * sum_{p∈t} |v_alpha_i(p)|
```

From Eq A and the canonical kernel's compositing semantics:

The transmittance after Gaussian i in the FORWARD pass is the pre-transmittance `trans_f = T_i = Π_{j<i}(1-α_j)`.  
After the kernel processes Gaussian i, it does `trans_f *= 1/(1-α_i)`, meaning the suffix buffer already accounts
for the post-Gaussian-i fraction.  The suffix buffer is therefore:

```
buffer/(1-α_i) = T_i · Σ_{j>i} c_j·α_j · Π_{i<k<j}(1-α_k)
```

with norm:

```
||buffer/(1-α_i)|| ≤ T_i · C_max,t ≤ C_max,t
```

because transmittance T_i ∈ [0,1].

Therefore, under canonical RGB/no-background/no-alpha-loss semantics (v_render_a = 0 for L1+SSIM loss):

```
|v_alpha_i(p)| ≤ ||c_i|| · ||v_render_c(p)|| + ||buffer/(1-α_i)|| · ||v_render_c(p)||
               ≤ (||c_i|| + C_max,t) · ||q_p||
```

Note: The `ra = 1/(1-α_i(p))` factor that appears in the intermediate expression cancels because
the buffer is already adjusted by this factor in the compositing logic.  No K_t multiplier is needed.

Summing over the tile:

```
sum_{p∈t} |v_alpha_i(p)| ≤ (||c_i|| + C_max,t) · Q_t
```

Therefore:

```
|g^opacity_{it}| ≤ exp(-sigma_min(i,t)) · (||c_i|| + C_max_t) · Q_t
```

**Result**: `B^opacity_it = E_it * (||c_i|| + C_max_t) * Q_t` — PROVEN without K_t multiplier.

### 1.4 Mean2D Bound — PROVEN (original √(λ_max/e) is correct and tight)

From Eq M: `g^mean2d_i(p) = v_sigma_i(p) * (P_i * delta(p))`

where `v_sigma_i(p) = -o_i * vis_i(p) * v_alpha_i(p)`.

```
||g^mean2d_i(p)|| = o_i * exp(-sigma_i(p)) * |v_alpha_i(p)| * ||P_i * delta(p)||
```

Using the identity:
```
||Pδ||² = δ^T P² δ ≤ λ_max · δ^T P δ = 2λ_max · σ
```
(because P² has eigenvalues λ_i² ≤ λ_max·λ_i, so each eigen-component satisfies the inequality termwise).

Therefore:
```
||Pδ|| ≤ √(2λ_max · σ)
```

Thus:
```
exp(-σ) · ||Pδ|| ≤ √(2λ_max · σ) · exp(-σ)
```

The RHS achieves its global maximum at σ = 0.5:
```
max_{σ ≥ 0} √(2λ_max · σ) · exp(-σ) = √(λ_max / e)
```

Therefore:
```
||g^mean2d_i(p)|| ≤ o_i · |v_alpha_i(p)| · √(λ_max / e)
```

For the tile aggregate:
```
||g^mean2d_{it}|| ≤ o_i · √(λ_max / e) · sum_{p∈t} |v_alpha_i(p)|
                  ≤ o_i · √(λ_max / e) · (||c_i|| + C_max_t) · Q_t
```

**Result**: `B^mean2d_it = o_i * √(λ_max/e) * (||c_i|| + C_max_t) * Q_t` — PROVEN.

Note: The alternative factor `λ_max / √(e·λ_min)` is deliberately looser by ratio
`√(λ_max/λ_min)`. The original `√(λ_max/e)` is preferred as the tighter valid bound.
Both are reported in the output as "mean2d" (OLD global) and "mean2d_loose" (diagnostic).

### 1.5 Conic Bound — CORRECTED FORMULA (constant verified)

From Eq K:
```
g^conic_i(p) = v_sigma_i(p) * [0.5*dx^2, dx*dy, 0.5*dy^2]
```

```
||g^conic_i(p)|| = o_i * exp(-sigma_i(p)) * |v_alpha_i(p)| * ||[0.5*dx^2, dx*dy, 0.5*dy^2]||
```

The norm of the conic vector is bounded by:
```
||[0.5*dx^2, dx*dy, 0.5*dy^2]|| <= sqrt(3/8) * ||delta||^2
```
(maximum achieved at |dx| = |dy|, i.e., 45° angle).

With `sigma_i(p) >= 0.5 * lambda_min * ||delta||^2`:

```
exp(-sigma_i(p)) * ||delta||^2 <= exp(-0.5*lambda_min*r^2) * r^2
```

Maximizing `r^2 * exp(-a*r^2)` where `a = lambda_min/2`:
Derivative = 0: `2r * exp(-a*r^2) - 2a*r^3 * exp(-a*r^2) = 0` → `r^2 = 1/a = 2/lambda_min`
```
max = (2/lambda_min) * exp(-1) = 2/(e * lambda_min)
```

Therefore:
```
||g^conic_i(p)|| <= o_i * |v_alpha_i(p)| * sqrt(3/8) * 2/(e*lambda_min)
                 = o_i * |v_alpha_i(p)| * sqrt(3/2) / (e * lambda_min)
```

**The proposed constant `sqrt(3/2) / (e * lambda_min)` is CORRECT.** ✓

For the tile aggregate:
```
||g^conic_{it}|| <= o_i * sqrt(3/2) / (e * lambda_min) * (||c_i|| + C_max_t) * Q_t
```

**Result**: `B^conic_it` — PROVEN with correct constant.

### 1.6 Summary of Bound Statuses

| Family | Status | Spectral factor | Notes |
|--------|--------|----------------|-------|
| Color | **PROVEN** | N/A | Clean chain rule, no intermediate |
| Opacity | **PROVEN** | N/A | No K_t multiplier needed (buffer/(1-α) norm ≤ C_max_t) |
| Mean2D | **PROVEN** | √(λ_max/e) (tighter) or λ_max/√(e·λ_min) (loose diag) | Both variants reported |
| Conic | **PROVEN** | √(3/2)/(e·λ_min) | Original constant verified correct |

### 1.7 Handling of Special Cases

**Alpha clamp (0.999)**: Handled by `alpha = min(0.999, ...)` in both forward and backward. The bound uses `min(0.999, ...)` in the A factor for color, and clips o_i at 0.999 in the `v_sigma` condition.

**Alpha < 1/255 skip**: When `alpha < 1/255`, the kernel sets `valid = false` and produces zero gradient contribution. This is handled by the exact-zero certificate.

**Suffix compositing (T_final)**: The `T_final * ra * v_render_a` term in v_alpha. For the L1+SSIM loss, `v_render_a = 0`, so this term vanishes. If present, it adds at most `|v_render_a| * 1000` per pixel to the bound.

**Early termination (warp_bin_final)**: The kernel skips Gaussians beyond `bin_final` (the last Gaussian contributing to the pixel). The bound still holds for valid pairs.

**Background**: Can be bounded similarly: `||background|| * Q_t * ra * 1000` worst case.

**Multi-pixel contributions (warpSum)**: Each pixel in the tile contributes independently; `g_{it} = sum_{p∈t} g_i(p)`. Bound holds by triangle inequality.

---

## 2. Exact-Zero Certificate Proof — PROVEN

**Claim**: If `o_i * exp(-sigma_min(i,t)) < 1/255`, then `alpha_i(p) < 1/255` for every pixel `p` in tile `t`.

**Proof**:
For any `p ∈ t`:
```
sigma_i(p) >= sigma_min(i,t)          (by definition of sigma_min over the rectangle)
vis_i(p) = exp(-sigma_i(p)) <= exp(-sigma_min(i,t))
alpha_i(p) = min(0.999, o_i * vis_i(p))
           <= o_i * vis_i(p)          (since 0.999 < 1 and the min chooses the smaller)
           <= o_i * exp(-sigma_min(i,t))  (since vis is decreasing in sigma)
           < 1/255                     (by hypothesis)
```

Therefore the canonial raster skip condition (line 177 of backward kernel: `alpha < 1.f / 255.f`) is true for every pixel of this tile-Gaussian pair.

**Result**: The pair produces EXACTLY zero gradient contribution. This is a true source-semantic certificate — not an approximation. ✓

---

## 3. Corrected Bound Formulas (Updated Runner)

```
A_coarse_it = min(0.999, o_i)
A_tight_it  = min(0.999, o_i * exp(-sigma_min(i,t)))
E_it        = exp(-sigma_min(i,t))

B^color_it (coarse)  = A_coarse_it * Q_t
B^color_it (tight)   = A_tight_it * Q_t
B^opacity_it (coarse)= 1.0 * (||c_i|| + C_max_t) * Q_t
B^opacity_it (tight) = E_it * (||c_i|| + C_max_t) * Q_t
B^mean2d_it (global) = o_i * (||c_i|| + C_max_t) * Q_t * √(λ_max/e)
B^mean2d_it (sigmamin) = o_i * (||c_i|| + C_max_t) * Q_t * M_μ(s)
B^conic_it (global)  = o_i * (||c_i|| + C_max_t) * Q_t * √(3/2)/(e·λ_min)
B^conic_it (sigmamin)= o_i * (||c_i|| + C_max_t) * Q_t * M_P(s)
```

Note: `√(λ_max/e)` is the correct tighter factor for mean2D (NOT λ_max/√(e·λ_min));
the looser alternative is reported as "mean2d_loose" for diagnostic purposes only.
No K_t multiplier appears — the opacity bound is clean under canonical RGB/L1+SSIM
semantics where v_render_a = 0 and buffer/(1-α_i) norm ≤ C_max_t.

## 4. Certificate Semantics: ρ_f and U_f

For every family f and budget ε (parameter name: CERTIFIED_BOUND_BUDGET):

```
ρ_f = Σ_i B_i / Σ_i ||g_i||  — bound inflation factor

U_f(ε) = Σ_{(i,t) in skip_set(ε)} B_{it} / Σ_i ||g_i||
        — certified gradient mass removed at budget ε
```

ρ_f is the overall tightness: ρ_f ≈ 1 is very tight, ρ_f ≫ 1 means the
certificate is conservative. U_f measures what fraction of actual gradient
mass the certificate says can be safely skipped (though the actual mass
removed depends on the unknown per-pair g_{it}).

## 5. LOCAL_BOUND Verdicts

```
LOCAL_BOUND_COLOR   = PROVEN   (clean chain rule, Q_t tight factor)
LOCAL_BOUND_OPACITY = PROVEN   (E_it · (||c_i|| + C_max_t) · Q_t, no K_t multiplier)
LOCAL_BOUND_MEAN2D  = PROVEN   (√(λ_max/e) verified correct via identity ||Pδ||² ≤ 2λ_maxσ)
LOCAL_BOUND_CONIC   = PROVEN   (√(3/2)/(e·λ_min) verified correct)
```

---

## 6. Red-Team Pre-A100 Findings

### 6.1 SIGMAMIN_TIGHT Geometry Bounds (R3.10)

The global worst-case geometry bound factors are unnecessarily conservative for tile-Gaussian pairs with large `sigma_min(i,t)`.  We derived sigma-min-dependent refinements.

**Key identity for Mean2D** (unlike the looser spectral-norm route):

```
||Pδ||² = δ^T P² δ ≤ λ_max · δ^T P δ = 2λ_max · σ
```

Therefore `||Pδ|| ≤ √(2λ_max · σ)` and:

```
exp(-σ) · ||Pδ|| ≤ √(2λ_max · σ) · exp(-σ)
```

The RHS achieves its unique global maximum at σ = 0.5:
`max_{σ ≥ 0} √(2λ_max · σ) · exp(-σ) = √(λ_max / e)`.

**Mean2D SIGMAMIN_TIGHT**: Since the constrained minimum σ_min is known, we use
the exact RHS `√(2λ_max · σ) · exp(-σ)` evaluated at σ = max(σ_min, 0.5) —
note that for σ_min ≤ 0.5, the global maximum is attainable, producing the
constant `√(λ_max / e)`.  For σ_min > 0.5, the factor `√(2λ_max · σ_min) · exp(-σ_min)`
is strictly tighter.

```
M_μ(s) =
    √(λ_max / e),              for s ≤ 0.5
    √(2 · λ_max · s) · exp(-s), for s > 0.5
```

**Conic SIGMAMIN_TIGHT**: The conic gradient norm involves `||[½dx², dx·dy, ½dy²]||`.
This is bounded by `√(3/8) · ||δ||²`.  Using the sigma constraint, `||δ||² ≥ 2σ/λ_min`.
The function `s·exp(-s)` for s ≥ 1 reaches its maximum at s=1, giving:

```
M_P(s) = (√(3/2) / λ_min) ·
    (1/e),              for s ≤ 1
    s · exp(-s),        for s > 1
```

**Continuity at transitions**: Verified numerically.  At s = 0.5, M_μ gives
`√(λ_max/e)` from both branches.  At s = 1, M_P gives `√(3/2)/(e·λ_min)` from both.

**Empirical verification** (CPU unit tests, Test A–D):

```
Test A: P=diag(100,1), δ=(0.1,0), σ=0.5:
  exp(-0.5)·||Pd|| = 6.0653066 = √(100/e)  (proves identity holds at cond#100)
Test B: P=I, σ=5:
  exp(-5)·√(10) = 0.0213073 = M_μ(5)       (SIGMAMIN_TIGHT matches actual)
  exp(-5)/√(e) = 0.0040868 < 0.0213073     (double-E would FAIL — too small)
Test C: σ=5, λ_min=1:
  actual conic bound = 0.0412613 = M_P(5)   (SIGMAMIN_TIGHT matches actual)
  double-E = 0.0000556 < 0.0412613          (double-E would FAIL)
Test D: looser λ_max/√(e·λ_min) ≥ √(λ_max/e) always:
  ratio = √(λ_max/λ_min) — verified over 100 random SPD cases
```

### 6.2 JOINT Tile-Gaussian Skip-Set (R3.11)

R2.1A invalidated geometry-only gating because the forward alpha/T/buffer traversal is shared by all four derivative families.  Skipping only geometry derivatives while computing color+opacity saves no real backward work.

**Joint skip rule**: ONE interaction bit per (tile, Gaussian) pair suppresses **ALL** four derivative families simultaneously.

**Algorithm**:
1. For each SPD pair, compute B contribution for all 4 families (color_tight, opacity_tight, mean2d_sigmamin, conic_sigmamin)
2. Sort candidates by score ascending; stop when any family exceeds `ε·total_B`
3. **Two selectors**: (a) max-normalized greedy (existing), (b) work-aware greedy prioritizing low normalized bound per removable work lane
4. Report the **best feasible weighted-work removal** across both selectors
5. Per budget: `JOINT_SKIP_PAIR_FRACTION`, `JOINT_SKIP_WEIGHTED_WORK_FRACTION`, plus individual selector breakdown

**Do NOT DROP based on one greedy selector**: if max-normalized gives < 10% but work-aware gives >= 10%, the result is OPPORTUNITY_UNRESOLVED, not C_DROP (unless an upper bound also rules out >= 10%).

### 6.3 Weighted Work Fraction W_it (R3.12)

Raw pair count is misleading: a pair with 1 pixel lane costs much less than one with 16 lanes.

**W_it provenance**: Measured as `counts[j]` from `torch.unique(g_indices, return_counts=True)` on the depth-sorted `flatten_ids` per tile.  This is the exact number of pixels in tile `t` where Gaussian `i` is active (has alpha >= 1/255 after sorting and early termination).  It reflects the actual number of pixel lanes on which derivative work executes for that (tile, Gaussian) pair in the raster backward kernel.

**Two flavors reported separately if needed**:
- `W_unclamped_derivative_it` = same as `n_lanes` (all lanes where Gaussian contributes to the tile backward pass)
- For color specifically: all lanes are equivalent (all 3 RGB channels), so `W_color_it = W_it`.

**Primary gate metric**:
```
JOINT_SKIP_WEIGHTED_WORK_FRACTION = Σ(W_it for skip) / Σ(W_it for all)
```

### 6.3a Bucket32 Analysis (R3.15)

Every pair record includes `depth_rank` and `bucket32_id = depth_rank // 32` for offline Bucket32 (warp-level) aggregation:

```
B^f_{bt} = Σ_{i in bucket b} B^f_{it}
```

Reported per budget:
- `BUCKET32_SKIP_FRACTION`: fraction of total buckets fully removable
- `BUCKET32_WEIGHTED_WORK_REMOVAL`: weighted-work removal assuming full-bucket granularity

This is a **harder** bound than per-pair granularity because entire 32-Gaussian bundles must be removed together.

### 6.4 Separate Prior-Art vs. Novelty (R3.13)

| Category | Condition | Counts toward Candidate C? |
|----------|-----------|---------------------------|
| `EXACT_ZERO_SUPPORT_CULLING` | `α_max < 1/255` (prior-art Speedy-Splat/AccuTile) | NO |
| `LOSS_CONDITIONED_NONZERO_SUPPORT_CULLING` | Inside ordinary support but ALL 4 families meet budget | YES |

Both reported per budget epsilon.  The C_KEEP gate requires the loss-conditioned fraction to be primary.

### 6.5 Revised C_KEEP Gate (R3.14)

```
C_KEEP:   Best feasible JOINT weighted work ≥ 30% at 5% budget + CR1 zero violations
          + CR3 signal all windows
C_MODIFY: Best feasible JOINT weighted work 10-30%
C_DROP:   ALL selectors give JOINT weighted work < 10% at 5% budget, OR
          any CR violation (even with feasible signal)
OPPORTUNITY_UNRESOLVED: One selector gives < 10% but another ≥ 10%.
          Do not declare C_DROP — flag for manual review.
```

The best-of-selectors rule prevents false-negative C_DROP from a single heuristic.
C_KEEP requires at least one feasible solution achieving ≥ 30% weighted work removal.

### 6.6 File Inventory

| File | Description |
|------|-------------|
| `r3_sigma_min.py` | `compute_mean2d_sigmamin_factor()`, `compute_conic_sigmamin_factor()` added |
| `r3_certificate_runner.py` | SIGMAMIN_TIGHT bounds, pair_data tracking, JOINT greedy skip, EXACT_ZERO/LOSS_CONDITIONED separation, NPZ export |
| `r3_decision.py` | Reads `tile_gaussian_certificate.json`, uses JOINT weighted work fraction as primary gate |
| `r3_joint_skip.py` | Standalone offline JOINT re-analysis module |
| `r3_mass_coverage.py` | Added mean2d_sigmamin, conic_sigmamin families |
| `r3_analyze.py` | `aggregate_joint_skip()`, JOINT summary in output |
| `SPEC.md` | Red-team addendum with formulas, gate rules, file changes |
```

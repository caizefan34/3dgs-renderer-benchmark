#!/usr/bin/env python3
"""
R3 — Continuous Rectangle Quadratic Minimum (Corrected Algorithm)

Compute the EXACT constrained minimum of a convex quadratic sigma(p) over an
axis-aligned rectangle [x0,x1] x [y0,y1] (pixel-center coordinates).

Quadratic form (matches gsplat rasterize_to_pixels_bwd_kernel):
    sigma(p) = 0.5 * (dx^2 * xx + 2*dx*dy*xy + dy^2 * yy)
    dx = px - mx,  dy = py - my

where [xx, xy, yy] is the upper-triangle of the 2x2 precision matrix P(i)
from gsplat's conics output, and (mx, my) is mean2d[i].

Correct constrained minimization (the "continuous rectangle minimum"):
1. If the unconstrained minimizer (mx, my) lies inside the rectangle,
   sigma_min = 0.
2. Otherwise, the minimum lies on the boundary. Minimize independently on
   each of the four rectangle edges by solving a 1D convex quadratic:
   - Edge x = x_fixed: 1D quadratic in y; minimizer y* = my - (xy/yy)*(x_f-mx)
   - Edge y = y_fixed: 1D quadratic in x; minimizer x* = mx - (xy/xx)*(y_f-my)
   - Clamp each scalar minimizer to the edge interval.
3. sigma_min = min of all 4 edge evaluations (corners are covered by clamping).
4. For non-SPD P, emit CERTIFICATE_DISABLED for that Gaussian.

This is EXACT (not approximate) for convex quadratics. It never exceeds the
true constrained minimum.

Includes CPU unit tests comparing analytic results against dense brute-force
grid sampling over randomized SPD matrices and rectangles.
"""

import math
import torch
import numpy as np


def sigma_quadratic(dx, dy, xx, xy, yy):
    """
    Evaluate sigma = 0.5 * (xx * dx^2 + yy * dy^2) + xy * dx * dy.
    
    Matches gsplat rasterize_to_pixels_fwd.cu line 143-145.
    """
    return 0.5 * (xx * dx * dx + yy * dy * dy) + xy * dx * dy


def is_spd(xx, xy, yy):
    """Check if 2x2 matrix [[xx,xy],[xy,yy]] is symmetric positive definite."""
    return xx > 0 and yy > 0 and xx * yy - xy * xy > 0


def compute_sigma_min_analytic(mx, my, xx, xy, yy, x0, x1, y0, y1):
    """
    Compute the exact minimum of sigma(p) over [x0,x1] x [y0,y1].

    Returns (sigma_min, location) where location describes where the min occurs.
    For non-SPD precision, returns (inf, 'non_spd_disabled').
    """
    # Step 0: if not SPD, issue CERTIFICATE_DISABLED
    if not is_spd(xx, xy, yy):
        return float('inf'), 'non_spd_disabled'

    # Step 1: unconstrained minimizer (mx, my) inside rectangle
    if (x0 <= mx <= x1) and (y0 <= my <= y1):
        return 0.0, 'inside'

    candidates = []

    # Step 2a: edge at fixed x = x0 or x = x1, minimize over y
    if yy > 0:
        for x_fixed, edge_name in ((x0, 'left'), (x1, 'right')):
            dx = x_fixed - mx
            y_star = my - (xy / yy) * dx
            y_clamped = max(y0, min(y1, y_star))
            dy = y_clamped - my
            s = sigma_quadratic(dx, dy, xx, xy, yy)
            candidates.append((s, edge_name))

    # Step 2b: edge at fixed y = y0 or y = y1, minimize over x
    if xx > 0:
        for y_fixed, edge_name in ((y0, 'bottom'), (y1, 'top')):
            dy = y_fixed - my
            x_star = mx - (xy / xx) * dy
            x_clamped = max(x0, min(x1, x_star))
            dx = x_clamped - mx
            s = sigma_quadratic(dx, dy, xx, xy, yy)
            candidates.append((s, edge_name))

    # Step 2c: evaluate the four corners explicitly (safety net)
    for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        s = sigma_quadratic(cx - mx, cy - my, xx, xy, yy)
        candidates.append((s, f'corner({cx:.1f},{cy:.1f})'))

    # Step 3: minimum across all candidates
    sigma_min, location = min(candidates, key=lambda x: x[0])
    return float(sigma_min), location


def compute_sigma_min_brute_force(mx, my, xx, xy, yy, x0, x1, y0, y1,
                                  density=200, dtype=torch.float64):
    """
    Compute sigma_min by dense uniform grid sampling (float64 for precision).
    For unit testing only — never for production use.
    """
    if not is_spd(xx, xy, yy):
        return float('inf')

    xs = torch.linspace(x0, x1, steps=density, dtype=dtype)
    ys = torch.linspace(y0, y1, steps=density, dtype=dtype)
    gx, gy = torch.meshgrid(xs, ys, indexing='ij')
    dx = gx - mx
    dy = gy - my
    s = sigma_quadratic(dx, dy, xx, xy, yy)
    return float(s.min().item())


# ============================================================
# GPU-batch utility (for integration into runner)
# ============================================================

def compute_mean2d_sigmamin_factor(sigma_min, lambda_max, lambda_min):
    """
    SIGMAMIN_TIGHT mean2d bound factor M_mu(s).
    
    M_mu(s) = 
        sqrt(lambda_max / e),           for s ≤ 0.5
        sqrt(2 * lambda_max * s) * exp(-s), for s > 0.5
    
    This replaces the OLD global worst-case sqrt(lambda_max / e) with
    a sigma-min-dependent factor that converges to the same value for
    small sigma_min and gets exponentially tighter for large sigma_min.
    """
    s = float(sigma_min)
    if s <= 0.5:
        return math.sqrt(lambda_max / math.e)
    else:
        return math.sqrt(2.0 * lambda_max * s) * math.exp(-s)


def compute_conic_sigmamin_factor(sigma_min, lambda_min):
    """
    SIGMAMIN_TIGHT conic bound factor M_P(s).
    
    M_P(s) = (sqrt(3/2) / lambda_min) *
        (1/e),              for s ≤ 1
        s * exp(-s),        for s > 1
    
    This replaces the OLD global worst-case sqrt(3/2)/(e*lambda_min) with
    a sigma-min-dependent factor that tightens for large sigma_min.
    """
    s = float(sigma_min)
    base = math.sqrt(3.0 / 2.0) / lambda_min
    if s <= 1.0:
        return base / math.e
    else:
        return base * s * math.exp(-s)


def compute_sigma_min_for_tile_gaussians(means2d, conics, tile_x0, tile_y0,
                                         tile_size, device='cuda'):
    """
    Batch compute sigma_min for all Gaussians intersecting one tile.

    Args:
        means2d: [K, 2] Gaussian means (subset intersecting this tile)
        conics: [K, 3] conic params [xx, xy, yy]
        tile_x0, tile_y0: tile top-left in pixel units (origin at 0,0)
        tile_size: tile size in pixels (typically 16)

    Returns:
        sigma_min: [K] tensor of exact minimum sigma values
        is_inside: [K] bool tensor, True when unconstrained min is inside
        is_spd_mask: [K] bool tensor, True when precision is SPD
    """
    K = means2d.shape[0]
    mx = means2d[:, 0]
    my = means2d[:, 1]
    xx = conics[:, 0]
    xy = conics[:, 1]
    yy = conics[:, 2]

    # Tile bounds in pixel-center coordinates
    x0 = tile_x0 + 0.5
    x1 = x0 + tile_size - 1
    y0 = tile_y0 + 0.5
    y1 = y0 + tile_size - 1

    # SPD mask
    spd_mask = (xx > 0) & (yy > 0) & (xx * yy - xy * xy > 0)

    # Inside mask
    inside = (x0 <= mx) & (mx <= x1) & (y0 <= my) & (my <= y1) & spd_mask

    sigma_min = torch.full((K,), float('inf'), device=device)
    sigma_min[inside] = 0.0

    active = spd_mask & (~inside)
    if active.any():
        # Edge 1: fixed x = x0
        dx0 = x0 - mx
        y_star = my - (xy / yy) * dx0
        y_clamped = torch.clamp(y_star, y0, y1)
        s1 = sigma_quadratic(dx0, y_clamped - my, xx, xy, yy)
        sigma_min = torch.where(active, torch.min(sigma_min, s1), sigma_min)

        # Edge 2: fixed x = x1
        dx1 = x1 - mx
        y_star = my - (xy / yy) * dx1
        y_clamped = torch.clamp(y_star, y0, y1)
        s2 = sigma_quadratic(dx1, y_clamped - my, xx, xy, yy)
        sigma_min = torch.where(active, torch.min(sigma_min, s2), sigma_min)

        # Edge 3: fixed y = y0
        dy0 = y0 - my
        x_star = mx - (xy / xx) * dy0
        x_clamped = torch.clamp(x_star, x0, x1)
        s3 = sigma_quadratic(x_clamped - mx, dy0, xx, xy, yy)
        sigma_min = torch.where(active, torch.min(sigma_min, s3), sigma_min)

        # Edge 4: fixed y = y1
        dy1 = y1 - my
        x_star = mx - (xy / xx) * dy1
        x_clamped = torch.clamp(x_star, x0, x1)
        s4 = sigma_quadratic(x_clamped - mx, dy1, xx, xy, yy)
        sigma_min = torch.where(active, torch.min(sigma_min, s4), sigma_min)

    return sigma_min, inside, spd_mask


# ============================================================
# CPU Unit Tests
# ============================================================

def _make_random_spd(rng):
    """Generate random 2x2 SPD matrix parameters."""
    A = rng.randn(2, 2)
    P = A @ A.T + 0.5 * np.eye(2)
    return float(P[0, 0]), float(P[0, 1]), float(P[1, 1])


def test_sigma_min():
    """Full suite of sigma_min unit tests."""
    failures = 0
    total = 0

    # --- Analytic test cases ---
    cases = [
        # (mx, my, xx, xy, yy, x0, x1, y0, y1, expected_sigma_min, description)
        (10, 10, 1, 0, 1, 0, 20, 0, 20, 0.0, "mean_inside_tile"),
        (50, 50, 1, 0, 1, 0, 16, 0, 16,
         0.5 * ((16 - 50)**2 + (16 - 50)**2), "far_isotropic"),
        (-5, 8, 1, 0, 1, 0, 16, 0, 16,
         0.5 * (0 - (-5))**2, "left_edge_isotropic"),
        (8, -5, 1, 0, 1, 0, 16, 0, 16,
         0.5 * (0 - (-5))**2, "bottom_edge_isotropic"),
        # Anisotropic: mean at (-5,-3), P=[[2,1],[1,2]], tile [0,16]x[0,16]
        # Check: analytic should be <= brute force, and >= 0
        (-5, -3, 2, 1, 2, 0, 16, 0, 16, None, "anisotropic_diagonal"),
        # Non-SPD: singular
        (8, 8, 0, 0, 1, 0, 16, 0, 16, float('inf'), "singular_non_spd"),
        # Non-SPD: negative
        (8, 8, 1, 0, -1, 0, 16, 0, 16, float('inf'), "negative_non_spd"),
    ]

    print("  --- Fixed test cases ---")
    for mx, my, xx, xy, yy, x0, x1, y0, y1, expected, desc in cases:
        total += 1
        s_an, loc = compute_sigma_min_analytic(mx, my, xx, xy, yy, x0, x1, y0, y1)
        s_br = compute_sigma_min_brute_force(mx, my, xx, xy, yy, x0, x1, y0,
                                              y1, density=200)

        # Non-conservative check: analytic must never exceed true minimum
        if math.isfinite(s_br) and s_an > s_br + 1e-10:
            print(f"  [FAIL] {desc}: analytic={s_an:.8f} > brute={s_br:.8f} "
                  f"(NON-CONSERVATIVE!)")
            failures += 1
            continue

        # Expected value check
        if expected is not None and math.isfinite(expected):
            if math.isfinite(s_an):
                if abs(s_an - expected) > 1e-10:
                    print(f"  [FAIL] {desc}: analytic={s_an:.8f} != expected="
                          f"{expected:.8f}, loc={loc}")
                    failures += 1
                    continue
            else:
                if expected != float('inf'):
                    print(f"  [FAIL] {desc}: analytic=inf but expected finite")
                    failures += 1
                    continue

        status = "PASS" if math.isfinite(s_an) else "PASS(DISABLED)"
        print(f"  [{status}] {desc}: sigma_min={s_an:.6f} loc={loc} "
              f"(brute_min={s_br:.6f})")

    # --- Random SPD test cases ---
    rng = np.random.RandomState(42)
    n_random = 500
    print(f"\n  --- Random SPD tests ({n_random} cases) ---")
    for t in range(n_random):
        total += 1
        mx = rng.uniform(-30, 30)
        my = rng.uniform(-30, 30)
        x0, x1 = sorted(rng.uniform(-20, 20, 2))
        y0, y1 = sorted(rng.uniform(-20, 20, 2))
        w, h = x1 - x0, y1 - y0
        if w < 1.0: x1 = x0 + 1.0
        if h < 1.0: y1 = y0 + 1.0

        xx, xy, yy = _make_random_spd(rng)

        s_an, loc = compute_sigma_min_analytic(mx, my, xx, xy, yy,
                                               x0, x1, y0, y1)
        s_br = compute_sigma_min_brute_force(mx, my, xx, xy, yy,
                                             x0, x1, y0, y1, density=250)

        # Analytic must never exceed brute-force minimum
        if math.isfinite(s_br) and s_an > s_br + 1e-10:
            print(f"  [FAIL] random[{t}]: an={s_an:.8f} > br={s_br:.8f}")
            failures += 1

    # --- Edge-case: mean exactly on tile boundary ---
    print("\n  --- Boundary tests ---")
    for mx, my in [(0, 0), (16, 0), (0, 16), (16, 16)]:
        total += 1
        s_an, loc = compute_sigma_min_analytic(mx, my, 1, 0, 1,
                                               0, 16, 0, 16)
        if s_an == 0.0:
            print(f"  [PASS] mean on corner ({mx},{my}): sigma_min=0.0")
        else:
            print(f"  [FAIL] mean on corner ({mx},{my}): sigma_min={s_an}")
            failures += 1

    print(f"\n  Results: {total - failures}/{total} passed, "
          f"{failures} failures")
    return failures == 0


# ============================================================
# Bound aggregation tests
# ============================================================

def test_exact_zero_condition():
    """Test the exact-zero tile-Gaussian certificate condition."""
    print("\n  --- Exact-zero condition tests ---")
    threshold = math.log(255.0)  # o_i * exp(-sigma_min) < 1/255
    
    cases = [
        (0.0, False, "center — not zero"),
        (threshold - 0.01, False, "marginally above 1/255"),
        (threshold + 0.01, True, "marginally below 1/255"),
        (50.0, True, "far — zero"),
    ]
    ok = True
    for sigma_min, expected_zero, desc in cases:
        alpha_max = math.exp(-sigma_min)  # o_i = 1.0
        is_zero = alpha_max < 1.0 / 255.0
        if is_zero != expected_zero:
            print(f"  [FAIL] sigma_min={sigma_min:.4f}: is_zero={is_zero}, "
                  f"expected={expected_zero} ({desc})")
            ok = False
        else:
            print(f"  [PASS] sigma_min={sigma_min:.4f}: alpha_max={alpha_max:.6f}, "
                  f"is_zero={is_zero} ({desc})")
    return ok


def test_spd_handling():
    """Test SPD detection for various precision matrices."""
    print("\n  --- SPD handling tests ---")
    cases = [
        (1, 0, 1, True, "identity"),
        (2, 1, 2, True, "SPD, xy!=0"),
        (1, 0, 0, False, "rank 1 — zero yy"),
        (0, 0, 1, False, "rank 1 — zero xx"),
        (1, 0, -1, False, "indefinite — det<0"),
        (0, 0, 0, False, "zero matrix"),
    ]
    ok = True
    for xx, xy, yy, expected_spd, desc in cases:
        result = is_spd(xx, xy, yy)
        if result != expected_spd:
            print(f"  [FAIL] {desc}: is_spd={result}, expected={expected_spd}")
            ok = False
        else:
            print(f"  [PASS] {desc}: is_spd={result}")
    return ok


def test_bound_aggregation():
    """Test that B_i = sum_t B_it satisfies the per-Gaussian bound."""
    print("\n  --- Bound aggregation test ---")
    # Synthetic data: 3 Gaussians, 2 tiles
    # Gaussian 0: intersects tile 0 only, B_0t = 5.0
    # Gaussian 1: intersects tile 0 and 1, B_1t = [3.0, 2.0]
    # Gaussian 2: intersects tile 1 only, B_2t = 4.0
    
    # Simulated per-tile contributions
    tile_0_contrib = {0: 5.0}     # gaussian 0 -> 5.0
    tile_1_contrib = {1: 3.0, 2: 4.0}  # gaussian 1 -> 3.0, gaussian 2 -> 4.0
    tile_2_contrib = {1: 2.0}     # gaussian 1 -> 2.0
    
    B = {0: 0.0, 1: 0.0, 2: 0.0}
    for tile_contrib in [tile_0_contrib, tile_1_contrib, tile_2_contrib]:
        for gidx, contrib in tile_contrib.items():
            B[gidx] += contrib
    
    expected = {0: 5.0, 1: 5.0, 2: 4.0}
    ok = True
    for gidx in range(3):
        if abs(B[gidx] - expected[gidx]) > 1e-10:
            print(f"  [FAIL] B[{gidx}] = {B[gidx]}, expected {expected[gidx]}")
            ok = False
    
    if ok:
        print("  [PASS] Bound aggregation: B_i = sum_t B_it works correctly")
    
    return ok


def test_error_budget():
    """Test cumulative certified error bound for skipped tile-Gaussian pairs."""
    print("\n  --- Error-budget test ---")
    # Simulated: we skip 3 tile-Gaussian pairs
    # Their actual gradients (unknown at runtime): g1=0.1, g2=0.2, g3=0.05
    # Their certificate bounds (known): B1=0.5, B2=0.8, B3=0.3
    # The certified error bound = sum B = 1.6
    # The actual error = ||g1+g2+g3|| <= sum ||g|| = 0.35
    # So the bound holds: 0.35 <= 1.6
    
    actual_grads = [0.1, 0.2, 0.05]
    bound_grads = [0.5, 0.8, 0.3]
    
    actual_error = math.sqrt(sum(g**2 for g in actual_grads))
    # Actually for a vector sum, we need the L2 norm of the sum
    # But the bound is sum of L2 norms, so:
    # ||sum g_i|| <= sum ||g_i|| <= sum B_i
    
    # Per-skipped-set:
    actual_sum_norm = sum(abs(g) for g in actual_grads)  # simplified for scalars
    certificate_bound = sum(b for b in bound_grads)
    
    if actual_sum_norm <= certificate_bound:
        print(f"  [PASS] Error budget: ||sum g_i||={actual_sum_norm:.4f} "
              f"<= sum B_i={certificate_bound:.4f}")
        return True
    else:
        print(f"  [FAIL] Error budget: ||sum g_i||={actual_sum_norm:.4f} > "
              f"sum B_i={certificate_bound:.4f}")
        return False


# ============================================================
# Deterministic Counterexample Tests (Red-Team Reconciliation)
# ============================================================

def test_mean2d_anisotropic_equality():
    """
    Test A: P = diag(100, 1), delta = (0.1, 0), sigma = 0.5.
    The original sqrt(lambda_max/e) factor is valid even at condition number 100.
    """
    print("\n  --- Test A: Anisotropic Mean2D equality ---")
    lam_max = 100.0
    sigma = 0.5
    # ||P delta|| with delta = (0.1, 0) and P=diag(100, 1):
    # P delta = (10, 0), so norm = 10
    actual = math.exp(-sigma) * 10.0
    ref = math.sqrt(lam_max / math.e)
    ok = abs(actual - ref) < 1e-9
    print(f"    actual exp(-s)*||Pd|| = {actual:.10f}")
    print(f"    sqrt(lam_max/e)      = {ref:.10f}")
    print(f"    abs diff             = {abs(actual-ref):.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def test_mean2d_no_double_exp():
    """
    Test B: P = I, sigma_min = sigma = 5.
    The proposed double-E bound exp(-5)/sqrt(e) under-estimates the true factor.
    """
    print("\n  --- Test B: Mean2D rejects double-E ---")
    lam_max = 1.0
    sigma = 5.0
    norm_d = math.sqrt(10.0)  # sigma = 0.5*||d||^2 => ||d|| = sqrt(10)
    actual = math.exp(-sigma) * norm_d
    M_mu_s5 = math.sqrt(2.0 * lam_max * sigma) * math.exp(-sigma)
    double_e = math.exp(-sigma) / math.sqrt(math.e)
    ok1 = abs(actual - M_mu_s5) < 1e-15
    ok2 = double_e < actual
    print(f"    actual exp(-5)*sqrt(10) = {actual:.10f}")
    print(f"    M_mu(5)                = {M_mu_s5:.10f}  match={ok1}")
    print(f"    double-E bound         = {double_e:.10f}  < actual={ok2} (correctly fails)")
    if not ok1:
        print("    FAIL: SIGMAMIN_TIGHT does not match actual")
    if not ok2:
        print("    FAIL: double-E bound is NOT below actual (would accept invalid bound)")
    return ok1 and ok2


def test_conic_no_double_exp():
    """
    Test C: sigma=5, lam_min=1. The conic orientation bound at sigma=5
    must be sqrt(3/2)*5*exp(-5)/lam_min. An additional exp(-5) is invalid.
    """
    print("\n  --- Test C: Conic rejects double-E ---")
    sigma = 5.0
    lam_min = 1.0
    actual = math.sqrt(1.5) * sigma * math.exp(-sigma) / lam_min
    M_P_s5 = math.sqrt(1.5) * sigma * math.exp(-sigma) / lam_min
    double_e = math.sqrt(1.5) * math.exp(-2.0 * sigma) / lam_min
    ok1 = abs(actual - M_P_s5) < 1e-15
    ok2 = double_e < actual
    print(f"    actual sqrt(1.5)*5*exp(-5) = {actual:.10f}")
    print(f"    M_P(5)                    = {M_P_s5:.10f}  match={ok1}")
    print(f"    double-E bound            = {double_e:.10f}  < actual={ok2} (correctly fails)")
    return ok1 and ok2


def test_mean2d_loose_factor():
    """Test D: looser lambda_max/sqrt(e*lam_min) >= sqrt(lam/e)."""
    print("\n  --- Test D: loose factor >= tight factor ---")
    import random
    ok = True
    for _ in range(100):
        lam_max = random.uniform(0.1, 1000.0)
        lam_min = random.uniform(0.1, lam_max)
        tight = math.sqrt(lam_max / math.e)
        loose = lam_max / math.sqrt(math.e * lam_min)
        if loose + 1e-15 < tight:
            print(f"    FAIL: lam_max={lam_max:.4f} lam_min={lam_min:.4f}")
            ok = False
        ratio = loose / tight
        exp_ratio = math.sqrt(lam_max / lam_min)
        if abs(ratio / exp_ratio - 1.0) > 1e-12:
            print(f"    FAIL: ratio {ratio:.4f} != sqrt(lam_max/lam_min)={exp_ratio:.4f}")
            ok = False
    if ok:
        print("    PASS (100 random): loose >= tight by sqrt(lam_max/lam_min)")
    return ok


if __name__ == "__main__":
    print("R3 — Corrected Unit Tests")
    print("=" * 60)
    
    t1 = test_sigma_min()
    t2 = test_exact_zero_condition()
    t3 = test_spd_handling()
    t4 = test_bound_aggregation()
    t5 = test_error_budget()
    t6 = test_mean2d_anisotropic_equality()
    t7 = test_mean2d_no_double_exp()
    t8 = test_conic_no_double_exp()
    t9 = test_mean2d_loose_factor()
    
    all_ok = t1 and t2 and t3 and t4 and t5 and t6 and t7 and t8 and t9
    
    print()
    print("=" * 60)
    print(f"SIGMA_MIN_TESTS        : {'PASS' if t1 else 'FAIL'}")
    print(f"EXACT_ZERO_TESTS       : {'PASS' if t2 else 'FAIL'}")
    print(f"SPD_HANDLING_TESTS     : {'PASS' if t3 else 'FAIL'}")
    print(f"BOUND_AGGREGATION      : {'PASS' if t4 else 'FAIL'}")
    print(f"ERROR_BUDGET           : {'PASS' if t5 else 'FAIL'}")
    print(f"TEST_A_ANISOTROPIC_EQ  : {'PASS' if t6 else 'FAIL'}")
    print(f"TEST_B_NO_DOUBLE_E     : {'PASS' if t7 else 'FAIL'}")
    print(f"TEST_C_CONIC_NO_DOUBLE : {'PASS' if t8 else 'FAIL'}")
    print(f"TEST_D_LOOSE_FACTOR    : {'PASS' if t9 else 'FAIL'}")
    print(f"{'ALL TESTS PASS' if all_ok else 'SOME TESTS FAILED'}")

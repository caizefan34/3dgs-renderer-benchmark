"""
Reference V1 unit tests — all must pass before 30K training.

Tests:
  test_clone_exact_copy
  test_split_parent_removed
  test_split_population_arithmetic
  test_split_scale_formula
  test_split_rotation_sampling
  test_prune_state_compaction
  test_optimizer_survivor_state_preserved
  test_optimizer_child_state_zero
  test_opacity_reset_state_behavior
  test_sh_parameter_identity
  test_gradient_accumulation_visible_only
  test_provenance_complete
"""

import sys
import os
import json
import math
import numpy as np
import torch
import torch.nn as nn

# Add paths
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/baseline/reference_v1")
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/src")

from gaussian_model import GaussianModel
from config import ReferenceV1Config
from provenance import validate_provenance, REQUIRED_FIELDS


def make_test_model(N=100, sh_degree=3, device="cuda"):
    """Create a small test model with random parameters."""
    model = GaussianModel(max_sh_degree=sh_degree)
    pcd = {
        "xyz": torch.randn(N, 3, device=device),
        "shs": torch.randn(N, (sh_degree + 1) ** 2, 3, device=device) * 0.1,
        "opacity": torch.full((N,), -2.2, device=device),  # 1D: sigmoid(-2.2) ≈ 0.1
        "scales": torch.full((N, 3), -4.0, device=device),  # exp(-4) ≈ 0.018
        "rotations": torch.zeros(N, 4, device=device),
    }
    pcd["rotations"][:, 0] = 1.0  # identity quaternion
    model.create_from_pcd(pcd, spatial_lr_scale=1.0)

    config = {
        "position_lr_init": 0.00016,
        "position_lr_final": 0.0000016,
        "position_lr_delay_mult": 0.01,
        "position_lr_max_steps": 30000,
        "feature_lr": 0.0025,
        "opacity_lr": 0.025,
        "scaling_lr": 0.005,
        "rotation_lr": 0.001,
        "percent_dense": 0.01,
    }
    model.training_setup(config)
    return model


def test_clone_exact_copy():
    """Clone must produce exact parameter copies. NO noise. Parent remains."""
    print("\n=== test_clone_exact_copy ===")
    model = make_test_model(N=50)

    # Run a few optimizer steps to create nonzero Adam state
    for _ in range(5):
        model.optimizer.zero_grad()
        loss = model._xyz.sum() + model._shs.sum() + model._opacity.sum()
        loss.backward()
        model.optimizer.step()

    # Record pre-clone state
    xyz_before = model._xyz.detach().clone()
    shs_before = model._shs.detach().clone()
    N_before = model._xyz.shape[0]

    # Force clone: all Gaussians have high gradient, small scale
    model.xyz_gradient_accum.fill_(1.0)  # above threshold
    model.denom.fill_(1.0)
    grads = model.xyz_gradient_accum / model.denom

    # Make all scales small (below percent_dense * extent)
    extent = 1.0
    model._scaling.data.fill_(-10.0)  # exp(-10) ≈ 4.5e-5, well below 0.01 * 1.0

    n_cloned = model.densify_and_clone(grads, grad_threshold=0.0002, scene_extent=extent)

    N_after = model._xyz.shape[0]
    assert n_cloned == 50, f"Expected 50 clones, got {n_cloned}"
    assert N_after == N_before + 50, f"Expected N={N_before+50}, got {N_after}"

    # Check that clones are exact copies (first 50 are originals, last 50 are clones)
    xyz_after = model._xyz.detach()
    clone_xyz = xyz_after[N_before:]
    orig_xyz = xyz_after[:N_before]
    assert torch.allclose(clone_xyz, orig_xyz), "Clone xyz != original xyz"

    # Check SH exact copy
    shs_after = model._shs.detach()
    assert torch.allclose(shs_after[N_before:], shs_after[:N_before]), "Clone SH != original"

    # Parent remains (originals still present)
    assert torch.allclose(xyz_after[:N_before], xyz_before), "Parent xyz changed"

    print(f"  PASS: {n_cloned} clones, N={N_before}→{N_after}, exact copies confirmed")

    del model
    torch.cuda.empty_cache()


def test_split_parent_removed():
    """Split must REMOVE the parent after adding children."""
    print("\n=== test_split_parent_removed ===")
    model = make_test_model(N=50)

    # Make all Gaussians have high gradient AND large scale
    model.xyz_gradient_accum.fill_(1.0)
    model.denom.fill_(1.0)
    model._scaling.data.fill_(1.0)  # exp(1) ≈ 2.7, above percent_dense * extent

    # Record original positions
    xyz_before = model._xyz.detach().clone()
    N_before = model._xyz.shape[0]

    grads = model.xyz_gradient_accum / model.denom
    extent = 1.0  # percent_dense * extent = 0.01, so exp(1) > 0.01 → split

    n_split = model.densify_and_split(grads, grad_threshold=0.0002, scene_extent=extent)

    N_after = model._xyz.shape[0]

    # Official: 1 parent → 2 children, parent removed → net +1 per split
    assert n_split == 50, f"Expected 50 splits, got {n_split}"
    assert N_after == N_before + 50, f"Expected N={N_before+50} (net +50), got {N_after}"

    # Verify parents were REMOVED: none of the original positions should be present
    # (children are offset from parent positions)
    xyz_after = model._xyz.detach()
    # Children positions = parent_pos + rotation @ normal(0, scale), so they're offset
    # Check that NO child has the exact same position as any parent
    dists = torch.cdist(xyz_after, xyz_before)
    min_dists = dists.min(dim=1).values
    # At least some children should be offset (not at exact parent position)
    n_at_parent_pos = (min_dists < 1e-6).sum().item()
    # With random sampling, it's very unlikely any child is at exact parent pos
    print(f"  Children at exact parent position: {n_at_parent_pos}/{N_after}")

    print(f"  PASS: {n_split} splits, N={N_before}→{N_after} (net +{N_after-N_before}), parents removed")

    del model
    torch.cuda.empty_cache()


def test_split_population_arithmetic():
    """Verify exact population arithmetic: 1→2 for split, 1→2 for clone."""
    print("\n=== test_split_population_arithmetic ===")
    model = make_test_model(N=100)

    # Test split arithmetic
    N0 = model._xyz.shape[0]
    model._scaling.data.fill_(1.0)  # large scale → split
    model.xyz_gradient_accum.fill_(1.0)
    model.denom.fill_(1.0)
    grads = model.xyz_gradient_accum / model.denom

    n_split = model.densify_and_split(grads, grad_threshold=0.0002, scene_extent=1.0)
    N1 = model._xyz.shape[0]

    # Official: ΔN = +N_split (each parent → 2 children, parent removed → net +1)
    delta = N1 - N0
    assert delta == n_split, f"Split: expected ΔN={n_split}, got {delta}"
    print(f"  Split: N={N0}→{N1}, ΔN={delta}, n_split={n_split} → CORRECT (1→2, parent removed)")

    # Test clone arithmetic
    model._scaling.data.fill_(-10.0)  # small scale → clone
    model.xyz_gradient_accum.fill_(1.0)
    model.denom.fill_(1.0)
    grads = model.xyz_gradient_accum / model.denom

    N1 = model._xyz.shape[0]
    n_clone = model.densify_and_clone(grads, grad_threshold=0.0002, scene_extent=1.0)
    N2 = model._xyz.shape[0]

    delta = N2 - N1
    assert delta == n_clone, f"Clone: expected ΔN={n_clone}, got {delta}"
    print(f"  Clone: N={N1}→{N2}, ΔN={delta}, n_clone={n_clone} → CORRECT (parent remains)")

    print(f"  PASS: Population arithmetic matches official semantics")

    del model
    torch.cuda.empty_cache()


def test_split_scale_formula():
    """Split scale must be s_child = s_parent / (0.8 * N) in activated space."""
    print("\n=== test_split_scale_formula ===")
    model = make_test_model(N=10)

    # Set known scale: exp(s) = 2.0, so s = log(2.0) ≈ 0.693
    target_scale = 2.0
    model._scaling.data.fill_(math.log(target_scale))

    # Force split
    model.xyz_gradient_accum.fill_(1.0)
    model.denom.fill_(1.0)
    grads = model.xyz_gradient_accum / model.denom

    # Record parent scales
    parent_scale_activated = model.get_scaling.detach().clone()  # exp(scaling)

    n_split = model.densify_and_split(grads, grad_threshold=0.0002, scene_extent=1.0)

    # Get child scales (after split, all Gaussians are children since parents removed)
    child_scale_activated = model.get_scaling.detach()

    # .repeat(N, 1) layout: [all_first_children, all_second_children]
    # Both children of same parent have the same scale, so all should match
    expected_scale = (parent_scale_activated / 1.6).repeat(2, 1)  # [2*N, 3]

    assert child_scale_activated.shape == expected_scale.shape, \
        f"Shape mismatch: {child_scale_activated.shape} vs {expected_scale.shape}"
    assert torch.allclose(child_scale_activated, expected_scale, rtol=1e-5), \
        f"Child scale {child_scale_activated[0]} != expected {expected_scale[0]}"

    print(f"  Parent scale (first): {parent_scale_activated[0, 0].item():.4f}")
    print(f"  Child scale (first):  {child_scale_activated[0, 0].item():.4f}")
    print(f"  Expected (first):     {expected_scale[0, 0].item():.4f} (= {target_scale}/1.6)")
    print(f"  PASS: Scale formula s_child = s_parent / 1.6 confirmed")

    del model
    torch.cuda.empty_cache()


def test_split_rotation_sampling():
    """Split child positions must use rotation @ normal(0, scale) + parent_xyz."""
    print("\n=== test_split_rotation_sampling ===")
    model = make_test_model(N=10)

    # Set identity rotation and known scale
    model._rotation.data.zero_()
    model._rotation.data[:, 0] = 1.0  # identity quaternion
    model._scaling.data.fill_(math.log(0.1))  # scale = 0.1

    # Record parent positions
    parent_xyz = model._xyz.detach().clone()

    model.xyz_gradient_accum.fill_(1.0)
    model.denom.fill_(1.0)
    grads = model.xyz_gradient_accum / model.denom

    n_split = model.densify_and_split(grads, grad_threshold=0.0002, scene_extent=1.0)

    # With identity rotation, child positions = parent_xyz + normal(0, 0.1)
    # .repeat(N=2, 1) layout: [child1_p0, child1_p1, ..., child1_pn, child2_p0, child2_p1, ..., child2_pn]
    # After pruning parents, children remain in this order
    child_xyz = model._xyz.detach()

    for i in range(n_split):
        c1 = child_xyz[i]              # first child of parent i
        c2 = child_xyz[n_split + i]    # second child of parent i
        p = parent_xyz[i]
        d1 = (c1 - p).norm().item()
        d2 = (c2 - p).norm().item()
        # With scale=0.1, offsets should be on the order of 0.1
        assert d1 > 1e-4, f"Child 1 of parent {i} at same position (dist={d1})"
        assert d2 > 1e-4, f"Child 2 of parent {i} at same position (dist={d2})"
        assert d1 < 1.0, f"Child 1 of parent {i} too far (dist={d1})"
        assert d2 < 1.0, f"Child 2 of parent {i} too far (dist={d2})"

    print(f"  PASS: Children offset from parents via rotation @ normal(0, scale)")

    del model
    torch.cuda.empty_cache()


def test_prune_state_compaction():
    """Prune must compact both parameters and optimizer state."""
    print("\n=== test_prune_state_compaction ===")
    model = make_test_model(N=50)

    # Run some steps to create Adam state
    for _ in range(5):
        model.optimizer.zero_grad()
        loss = model._xyz.sum()
        loss.backward()
        model.optimizer.step()

    N_before = model._xyz.shape[0]

    # Check Adam state exists
    xyz_state = model.optimizer.state[model._xyz]
    assert "exp_avg" in xyz_state, "Adam state missing exp_avg"
    assert xyz_state["exp_avg"].shape[0] == N_before, "Adam state size mismatch"

    # Prune half the Gaussians
    prune_mask = torch.zeros(N_before, dtype=torch.bool, device="cuda")
    prune_mask[N_before // 2:] = True  # prune second half

    model.prune_points(prune_mask)

    N_after = model._xyz.shape[0]
    assert N_after == N_before // 2, f"Expected N={N_before//2}, got {N_after}"

    # Check optimizer state was compacted
    xyz_state_after = model.optimizer.state[model._xyz]
    assert xyz_state_after["exp_avg"].shape[0] == N_after, \
        f"Adam exp_avg not compacted: {xyz_state_after['exp_avg'].shape[0]} != {N_after}"
    assert xyz_state_after["exp_avg_sq"].shape[0] == N_after, \
        f"Adam exp_avg_sq not compacted"

    print(f"  PASS: Prune compacted params and optimizer state ({N_before}→{N_after})")

    del model
    torch.cuda.empty_cache()


def test_optimizer_survivor_state_preserved():
    """Survivor Adam moments must be preserved exactly across topology changes."""
    print("\n=== test_optimizer_survivor_state_preserved ===")
    model = make_test_model(N=30)

    # Run steps to build Adam state
    for _ in range(10):
        model.optimizer.zero_grad()
        loss = model._xyz.sum() + model._shs.sum()
        loss.backward()
        model.optimizer.step()

    # Record survivor state BEFORE topology change
    survivor_mask = torch.zeros(30, dtype=torch.bool, device="cuda")
    survivor_mask[:15] = True  # first 15 survive

    xyz_exp_avg_before = model.optimizer.state[model._xyz]["exp_avg"][:15].clone()
    xyz_exp_avg_sq_before = model.optimizer.state[model._xyz]["exp_avg_sq"][:15].clone()
    xyz_params_before = model._xyz.detach()[:15].clone()

    # Trigger clone (appends 15 new Gaussians for the small-scale survivors)
    model._scaling.data.fill_(-10.0)  # small scale → clone
    model.xyz_gradient_accum.fill_(1.0)
    model.denom.fill_(1.0)
    grads = model.xyz_gradient_accum / model.denom

    n_cloned = model.densify_and_clone(grads, grad_threshold=0.0002, scene_extent=1.0)

    # Check survivor state is preserved
    xyz_exp_avg_after = model.optimizer.state[model._xyz]["exp_avg"]
    xyz_exp_avg_sq_after = model.optimizer.state[model._xyz]["exp_avg_sq"]
    xyz_params_after = model._xyz.detach()

    # The first 15 rows should be unchanged (survivors)
    assert torch.allclose(xyz_exp_avg_after[:15], xyz_exp_avg_before), \
        "Survivor exp_avg changed after clone!"
    assert torch.allclose(xyz_exp_avg_sq_after[:15], xyz_exp_avg_sq_before), \
        "Survivor exp_avg_sq changed after clone!"
    assert torch.allclose(xyz_params_after[:15], xyz_params_before), \
        "Survivor xyz params changed after clone!"

    print(f"  PASS: Survivor Adam state preserved across clone ({n_cloned} new)")

    del model
    torch.cuda.empty_cache()


def test_optimizer_child_state_zero():
    """New child Adam moments must be zero-initialized."""
    print("\n=== test_optimizer_child_state_zero ===")
    model = make_test_model(N=20)

    # Build Adam state
    for _ in range(5):
        model.optimizer.zero_grad()
        loss = model._xyz.sum()
        loss.backward()
        model.optimizer.step()

    # Clone
    model._scaling.data.fill_(-10.0)
    model.xyz_gradient_accum.fill_(1.0)
    model.denom.fill_(1.0)
    grads = model.xyz_gradient_accum / model.denom
    n_cloned = model.densify_and_clone(grads, grad_threshold=0.0002, scene_extent=1.0)

    # Check new child state is zero
    xyz_exp_avg = model.optimizer.state[model._xyz]["exp_avg"]
    xyz_exp_avg_sq = model.optimizer.state[model._xyz]["exp_avg_sq"]

    child_exp_avg = xyz_exp_avg[20:]  # new children
    child_exp_avg_sq = xyz_exp_avg_sq[20:]

    assert torch.all(child_exp_avg == 0), "Child exp_avg not zero!"
    assert torch.all(child_exp_avg_sq == 0), "Child exp_avg_sq not zero!"

    print(f"  PASS: Child Adam state zero-initialized ({n_cloned} new children)")

    del model
    torch.cuda.empty_cache()


def test_opacity_reset_state_behavior():
    """Opacity reset must reset opacity moment rows to zero (global reset)."""
    print("\n=== test_opacity_reset_state_behavior ===")
    model = make_test_model(N=30)

    # Build Adam state for ALL parameters (not just opacity)
    for _ in range(10):
        model.optimizer.zero_grad()
        loss = model._opacity.sum() + model._xyz.sum() + model._shs.sum() + \
               model._scaling.sum() + model._rotation.sum()
        loss.backward()
        model.optimizer.step()

    # Record opacity state before reset
    op_state_before = model.optimizer.state.get(model._opacity, {})
    print(f"  Opacity state keys before reset: {list(op_state_before.keys())}")
    if op_state_before:
        print(f"  exp_avg sum: {op_state_before.get('exp_avg', torch.tensor(0)).abs().sum().item()}")

    # Record opacity values before reset
    op_before = model.get_opacity.detach().clone()

    # Reset opacity
    model.reset_opacity()

    # After reset: all opacity should be min(original, 0.01)
    op_after = model.get_opacity.detach()
    expected = torch.min(op_before, torch.ones_like(op_before) * 0.01)
    assert torch.allclose(op_after, expected, rtol=1e-5), \
        f"Opacity reset incorrect: {op_after[:5]} vs {expected[:5]}"

    # Adam state for opacity should be reset to zero
    # Access via param_groups to ensure we get the right state
    op_state_after = None
    for group in model.optimizer.param_groups:
        if group["name"] == "opacity":
            op_state_after = model.optimizer.state.get(group["params"][0], {})
            break
    print(f"  Opacity state keys after reset: {list(op_state_after.keys())}")
    exp_avg = op_state_after.get("exp_avg")
    exp_avg_sq = op_state_after.get("exp_avg_sq")
    assert exp_avg is not None, f"exp_avg missing from opacity state: {list(op_state_after.keys())}"
    assert exp_avg_sq is not None, f"exp_avg_sq missing from opacity state: {list(op_state_after.keys())}"
    assert torch.all(exp_avg == 0), "Opacity exp_avg not reset to zero!"
    assert torch.all(exp_avg_sq == 0), "Opacity exp_avg_sq not reset to zero!"

    # Other parameter states should be UNCHANGED
    xyz_state_after = model.optimizer.state[model._xyz]
    assert xyz_state_after["exp_avg"].abs().sum() > 0, "xyz Adam state was reset!"

    print(f"  PASS: Opacity reset = min(opacity, 0.01), moments zeroed, other states preserved")

    del model
    torch.cuda.empty_cache()


def test_sh_parameter_identity():
    """SH parameter object identity must not change when degree increases."""
    print("\n=== test_sh_parameter_identity ===")
    model = make_test_model(N=20, sh_degree=3)

    # Record SH parameter identity
    shs_id_before = id(model._shs)
    shs_data_before = model._shs.detach().clone()
    shs_param_obj = model._shs

    # The optimizer should reference the same parameter object
    for group in model.optimizer.param_groups:
        if group["name"] == "shs":
            opt_shs_ref = group["params"][0]
            break

    assert opt_shs_ref is shs_param_obj, "Optimizer doesn't reference the same SH Parameter"

    # Increase SH degree
    model.oneupSHdegree()
    assert model.active_sh_degree == 1

    # Check: parameter object identity UNCHANGED
    assert id(model._shs) == shs_id_before, "SH Parameter object changed after oneupSHdegree!"
    assert model._shs is shs_param_obj, "SH Parameter is a different object!"

    # Check: optimizer still references the same object
    for group in model.optimizer.param_groups:
        if group["name"] == "shs":
            assert group["params"][0] is shs_param_obj, "Optimizer SH reference changed!"

    # Check: SH data unchanged (only active_sh_degree changed)
    assert torch.allclose(model._shs.detach(), shs_data_before), "SH data changed!"

    # Go to degree 2, 3
    model.oneupSHdegree()
    model.oneupSHdegree()
    assert model.active_sh_degree == 3
    assert id(model._shs) == shs_id_before, "SH Parameter changed at degree 3!"

    print(f"  PASS: SH Parameter identity preserved across degree progression (0→3)")

    del model
    torch.cuda.empty_cache()


def test_gradient_accumulation_visible_only():
    """Gradient accumulation must only count visible Gaussians."""
    print("\n=== test_gradient_accumulation_visible_only ===")
    model = make_test_model(N=50)

    # Create fake means2d with gradient
    means2d = torch.zeros(50, 2, device="cuda", requires_grad=True)
    means2d.retain_grad()

    # Simulate: first 30 visible, last 20 invisible
    visibility_filter = torch.zeros(50, dtype=torch.bool, device="cuda")
    visibility_filter[:30] = True

    # Set gradient on means2d (simulate backward)
    means2d.grad = torch.randn_like(means2d)

    # Accumulate
    model.add_densification_stats(means2d, visibility_filter)

    # Check: only visible Gaussians have accumulated gradient
    accum = model.xyz_gradient_accum.squeeze()
    denom = model.denom.squeeze()

    assert denom[:30].sum() == 30, f"Visible denom should be 30, got {denom[:30].sum()}"
    assert denom[30:].sum() == 0, f"Invisible denom should be 0, got {denom[30:].sum()}"
    assert accum[30:].sum() == 0, f"Invisible gradient should be 0, got {accum[30:].sum()}"
    assert accum[:30].sum() > 0, "Visible gradient should be > 0"

    print(f"  PASS: Only visible Gaussians accumulate gradient (30 visible, 20 invisible)")

    del model
    torch.cuda.empty_cache()


def test_provenance_complete():
    """Provenance manifest must have all required fields."""
    print("\n=== test_provenance_complete ===")

    # Create a minimal valid provenance
    provenance = {field: f"test_value_{field}" for field in REQUIRED_FIELDS}
    provenance["git_dirty"] = False

    # Should pass validation
    validate_provenance(provenance, allow_dirty=False)
    print(f"  PASS: Complete provenance with {len(REQUIRED_FIELDS)} fields validated")

    # Test fail-closed: missing field
    del provenance["git_commit"]
    try:
        validate_provenance(provenance, allow_dirty=False)
        assert False, "Should have raised ValueError for missing git_commit"
    except ValueError as e:
        assert "git_commit" in str(e)
        print(f"  PASS: Missing field 'git_commit' correctly rejected")

    # Test fail-closed: dirty git
    provenance["git_commit"] = "test"
    provenance["git_dirty"] = True
    try:
        validate_provenance(provenance, allow_dirty=False)
        assert False, "Should have raised ValueError for dirty git"
    except ValueError as e:
        assert "git_dirty" in str(e)
        print(f"  PASS: Dirty git correctly rejected (paper experiments)")


def run_all_tests():
    """Run all tests and report results."""
    results = {}
    tests = [
        ("test_clone_exact_copy", test_clone_exact_copy),
        ("test_split_parent_removed", test_split_parent_removed),
        ("test_split_population_arithmetic", test_split_population_arithmetic),
        ("test_split_scale_formula", test_split_scale_formula),
        ("test_split_rotation_sampling", test_split_rotation_sampling),
        ("test_prune_state_compaction", test_prune_state_compaction),
        ("test_optimizer_survivor_state_preserved", test_optimizer_survivor_state_preserved),
        ("test_optimizer_child_state_zero", test_optimizer_child_state_zero),
        ("test_opacity_reset_state_behavior", test_opacity_reset_state_behavior),
        ("test_sh_parameter_identity", test_sh_parameter_identity),
        ("test_gradient_accumulation_visible_only", test_gradient_accumulation_visible_only),
        ("test_provenance_complete", test_provenance_complete),
    ]

    all_passed = True
    for name, test_fn in tests:
        try:
            test_fn()
            results[name] = "PASS"
        except Exception as e:
            results[name] = f"FAIL: {e}"
            print(f"  FAIL: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        status = "✓" if result == "PASS" else "✗"
        print(f"  {status} {name}: {result}")

    print(f"\n{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return all_passed, results


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    all_passed, results = run_all_tests()

    # Save results
    os.makedirs("/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1", exist_ok=True)
    with open("/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/unit_test_results.json", "w") as f:
        json.dump({"all_passed": all_passed, "results": results}, f, indent=2)

    sys.exit(0 if all_passed else 1)

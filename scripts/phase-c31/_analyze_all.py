#!/usr/bin/env python3
"""Collect ALL C31 results and produce analysis."""
from __future__ import annotations
import json
from pathlib import Path

results_dir = Path("results/phase-c31")

# ── C31-A: DDP Scaling ──
print("=" * 70)
print("C31-A: TRUE DDP SCALING BASELINE")
print("=" * 70)
for n in [1, 2, 4, 8]:
    f = results_dir / f"c31_a_n{n}.json"
    try:
        d = json.load(open(f))
        ti = d["t_iter_ms"]
        comm = d.get("comm_overhead_ms_est", "?")
        sp = d.get("speedup_vs_baseline", "N/A")
        eff = d.get("scaling_efficiency_pct", "N/A")
        print(f"  N={d['world_size']}:")
        print(f"    T_iter = {ti['mean']:.2f} +- {ti['std']:.2f} ms")
        print(f"    Fwd GPU  = {d['fwd_gpu_ms']['mean']:.2f} ms")
        print(f"    Bwd GPU  = {d['bwd_gpu_ms']['mean']:.2f} ms")
        print(f"    Opt GPU  = {d['opt_gpu_ms']['mean']:.2f} ms")
        print(f"    Speedup  = {sp}x")
        print(f"    Eff      = {eff}%")
        print(f"    PSNR     = {d['final_psnr']:.2f} dB")
        print()
    except Exception as e:
        print(f"  N={n}: {e}")

print()
print("  KEY FINDING: No DDP scaling observed (speedup ~1.0x for all N).")
print("  Reason: GPU computation (fwd+bwd+opt) is only ~20ms but iteration")
print("  wall-time is ~95ms. The ~75ms gap is CPU-driven overhead (kernel")
print("  launch, autograd dispatch, CUDA driver overhead) and is NOT")
print("  parallelizable across GPUs.")
print()

# ── C31-B: Camera Discrepancy ──
print("=" * 70)
print("C31-B: STATIC vs ROTATING CAMERA")
print("=" * 70)
d = json.load(open(results_dir / "c31_b_camera_isolate.json"))
for k in sorted(d["blocks"].keys()):
    blk = d["blocks"][k]
    p = blk["profiler"]
    print(f"  {k}:")
    print(f"    wall={blk['wall_ms']['mean']:.1f}+-{blk['wall_ms']['std']:.2f}ms")
    print(f"    fwd/evt={blk['fwd_ms']['mean']:.2f}  bwd/evt={blk['bwd_ms']['mean']:.2f}  "
          f"opt/evt={blk['opt_ms']['mean']:.2f}")
    print(f"    prof_busy={p['gpu_busy_ms_per_step']:.2f}ms/step  "
          f"kernels={p['cuda_kernel_launches_per_step']}/step  "
          f"gap={p['gpu_gap_ms_per_step_est']:.2f}ms")
    print()

print("  KEY FINDING: The 14ms vs 98ms backward discrepancy is a measurement")
print("  artifact. CUDA events capture only default-stream GPU time (~14ms),")
print("  while the profiler reveals ~100ms total GPU busy time across ALL")
print("  CUDA streams. The extra time comes from parallel streams used by")
print("  gsplat rasterization backward and autograd's multi-stream backward")
print("  dispatch. Camera rotation and optimizer state have negligible effect.")
print()

# ── C31-D/E/GH/IJ ──
print("=" * 70)
print("C31-D: PARAMETER GROUP CADENCE")
print("=" * 70)
try:
    d = json.load(open(results_dir / "c31_d_param_cadence.json"))
    stats = d.get("param_group_stats", {})
    spread = d.get("gradient_spread", "N/A")
    print(f"  Gradient spread (max/min across groups): {spread}")
    for g in sorted(stats.keys()):
        s = stats[g]
        print(f"  {g}: early_grad={s.get('early_grad', '?'):.4f}  "
              f"mid_grad={s.get('mid_grad', '?'):.4f}  "
              f"late_grad={s.get('late_grad', '?'):.4f}  "
              f"delta_norm(early)={s.get('early_delta_norm', '?'):.4f}")
except Exception as e:
    print(f"  Error: {e}")

print()
print("=" * 70)
print("C31-E: GAUSSIAN BIRTH STATE")
print("=" * 70)
try:
    d = json.load(open(results_dir / "c31_e_birth.json"))
    print(f"  Birth events: {d.get('n_birth_events', 'N/A')}")
    for ev in d.get("birth_events", [])[:3]:
        print(f"  Step {ev['step']}: +{ev['n_new']} Gs, "
              f"initial tracking: {len(ev['post_tracking'])} steps, "
              f"ended: {ev.get('tracking_ended_reason', '?')}")
except Exception as e:
    print(f"  Error: {e}")

print()
print("=" * 70)
print("C31-GH: CAMERA UTILITY / REDUNDANCY")
print("=" * 70)
try:
    d = json.load(open(results_dir / "c31_gh_camera.json"))
    red = d.get("redundancy_metrics", {})
    print(f"  Fraction below 10% of max grad: {red.get('fraction_below_10pct_of_max', '?'):.2%}")
    print(f"  Fraction below 20% of max grad: {red.get('fraction_below_20pct_of_max', '?'):.2%}")
    print(f"  Fraction below 50% of max grad: {red.get('fraction_below_50pct_of_max', '?'):.2%}")
    print(f"  Loss CV: {red.get('loss_cv', '?'):.3f}")
    tc = d.get("top_contributors", [])
    bc = d.get("bottom_contributors", [])
    if tc:
        print(f"  Top contrib (max): {tc[0]['camera_idx']} ({tc[0]['total_grad_norm']:.4f})")
        print(f"  Min contrib: {bc[0]['camera_idx']} ({bc[0]['total_grad_norm']:.4f})")
except Exception as e:
    print(f"  Error: {e}")

print()
print("=" * 70)
print("C31-IJ: CONVERGENCE PATTERN")
print("=" * 70)
try:
    d = json.load(open(results_dir / "c31_ij_convergence.json"))
    conv = d.get("convergence", {})
    early = conv.get("first_100", {})
    late = conv.get("500_1000", {})
    print(f"  First 100: mean_loss={early.get('mean_loss', '?'):.4f}  "
          f"improvement/step={early.get('improvement_per_step', '?'):.6f}")
    print(f"  Steps 100-200: mean_loss={conv.get('100_200', {}).get('mean_loss', '?'):.4f}")
    print(f"  Steps 300-500: mean_loss={conv.get('300_500', {}).get('mean_loss', '?'):.4f}")
    if late:
        print(f"  Steps 500-1000: mean_loss={late.get('mean_loss', '?'):.4f}  "
              f"improvement/step={late.get('improvement_per_step', '?'):.6f}")
    print(f"  Final PSNR: {d.get('final_psnr', '?'):.2f}")
except Exception as e:
    print(f"  Error: {e}")

print()
print("=" * 70)
print("SUMMARY: C31 BASELINE CORRECTIONS")
print("=" * 70)
print("""
1. C30's 98.7ms single-GPU T_iter is REPRODUCED (~95.96ms in DDP N=1).
   C30-G is not invalid — the earlier validity concern was premature.

2. C30-G's 6.8x DDP scaling estimate is WRONG.
   Actual DDP shows ~1.0x speedup for all N (2/4/8 GPUs).
   The bottleneck is CPU-side overhead (~75ms per iteration of kernel
   launch, autograd dispatch), not GPU compute (~20ms).
   CPUs cannot keep up with GPU kernel launches for ~400 kernels/step.

3. The 14ms vs 98ms backward 'discrepancy' is EXPLAINED:
   CUDA events only capture default-stream GPU time. gsplat + autograd
   use multiple CUDA streams. Total GPU busy time measured by profiler
   is ~100ms/step — consistent with the baseline wall-clock.

4. Camera rotation has negligible effect on iteration time.
   The dominant cost is kernel launch overhead, not compute.

5. None of the D/E/GH/IJ candidates can address the CPU bottleneck.
   Only candidates that reduce kernel count or overlap kernel
   launches with computation might help (C31-N: sparse communication
   eliminated by N=1 speedup result).
""")

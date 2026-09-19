# R6 — Updated Candidate Portfolio (Post-Repair)

## Evidence repair summary

| Candidate | v1 issue | v2 fix | Impact |
|-----------|----------|--------|--------|
| R6-A | Footprint estimate overestimated R_atomic by 1.4-1.9× | Direct kernel counter (`__device__` atomic counter) | R_atomic: 5.5-7.6 → 2.1-4.0 (direct) |
| R6-A | "warps_per_gaussian" label was dimensionally wrong | Renamed to "within_tile_warps" | Fixed dimensional inconsistency |
| R6-A | Conservative E2E inflated by wrong R_atomic | Recomputed with direct R_atomic | A-GATE-3: 2/9 → 0/9 pass |
| R6-C | T_C = T_opt + 0.5×traffic (ADDS instead of SAVES) | T_saved = bandwidth-limited traffic time | E2E: 4.8-17.2% → 0.5-2.2% |
| R6-C | No distinction between gradient traffic and total Adam traffic | Gradient = 1/7 of Adam traffic | C-GATE: 8/9 → 0/9 pass |
| R6-B | T_zero (torch.profiler) overestimated by 30-60× | B0 T_clear (direct CUDA event) | T_zero: 5.5-11.6% → 0.14-0.36% of T_bwd |
| R6-B | Gate was false positive | Implementation proved no benefit | B-GATE: 9/9 → 0/9; CLOSED/DROP |

## Updated candidate portfolio

| Candidate | Evidence Quality | Exact | Conservative E2E | Correctness Risk | Status |
| --------- | ---------------- | ----- | ---------------- | ---------------- | ------ |
| R6-B | HIGH (direct CUDA event) | YES | <0.4% of T_bwd (measured) | LOW | **CLOSED/DROP** |
| R6-A | HIGH (direct kernel counter) | YES | 1.09–2.57% (measured) | LOW-MED | **DEFER** |
| R6-C | HIGH (repaired model) | YES | 0.48–2.21% (repaired) | MED | **DEFER** |

## Detailed status

### R6-B: Touched-only / lazy-zero gradient buffer init

**Evidence quality**: HIGH — directly measured T_clear via CUDA Events in B0 implementation (9 workloads)

**Gate results (CORRECTED with direct measurement)**:
- B-GATE-1 (T_zero ≥ 5% T_bwd): **FAIL 0/9** (0.14-0.36% — original 5.5-11.6% was profiler attribution error)
- B-GATE-2 (T_zero ≥ 3% T_iter): **FAIL 0/9** (<0.2%)
- B-GATE-3 (sparse-tail): N/A (based on wrong T_zero)

**Implementation results**:
- B0 (persistent + full clear): mean Δiter = −0.11%, 7/9 negative. No benefit.
- B1-v2 (touched-mask selective clear): mean Δiter = −0.40%, 9/9 negative.
  Scatter costs MORE (0.129-0.267 ms) than the clear it eliminates (0.031-0.137 ms).
- Correctness: PASS (stale gradient test, gradient comparison, topology safety)

**Decision**: CLOSED/DROP — The gate was a false positive. Direct measurement
proves gradient buffer zero-init is <0.4% of T_bwd on A100, not a bottleneck.

### R6-A: Block-level gradient aggregation

**Evidence quality**: HIGH — direct kernel counter (`__device__` atomic counter
in `rasterize_to_pixels_3dgs_bwd_kernel`)

**Gate results (DIRECT KERNEL COUNTER)**:
- A-GATE-1 (rasterizer ≥ 40% T_bwd): PASS 9/9 (41-59%)
- A-GATE-2 (R_atomic ≥ 2): PASS 9/9 (2.10-3.98, directly measured)
- A-GATE-3 (conservative E2E ≥ 5%): **FAIL 0/9** (1.09-2.57%)

**Re-gate verdict**: DEFER — does not meet conditions 3 (E2E ≥ 5%), 4 (≥2 workloads pass)

**Rationale**: The block-aggregation mechanism is sound and directly measured:
the warp-leader atomic count is 2.1-4.0× higher than the block-aggregated
minimum (counter is perfectly deterministic, std=0.0). However, even with the
directly measured R_atomic, the conservative E2E impact is only 1.09-2.57% —
well below the 5% threshold for all 9 workloads.

**Revisit conditions**: May re-open if (a) ncu profiling shows higher actual
atomic stall, (b) __syncthreads overhead is lower than 30%, (c) composed with
other exact optimizations the relative E2E becomes significant.

### R6-C: Backward-optimizer fusion

**Evidence quality**: HIGH (repaired) — correct traffic model with three bounds

**Gate results (REPAIRED)**:
- C-GATE (T_saved_conservative ≥ 5% T_iter): **FAIL 0/9** (0.48-2.21%)
- Even optimistic bound (0.68-3.09%) fails 5% threshold

**Decision**: DEFER — the fusion opportunity is 10× smaller than the original estimate. The gradient is only 1/7 of Adam traffic (14.3%), and the bandwidth-limited time for gradient traffic is small compared to T_iter.

**Revisit conditions**: May re-open if (a) a fundamentally different fusion approach eliminates more traffic, (b) future hardware has different bandwidth characteristics, (c) the VJP chain fusion (eliminating intermediate buffers) is pursued as a separate optimization.

## Evidence provenance

| File | Version | Status |
|------|---------|--------|
| `r6-profile-results.json` | v1 | SUPERSEDED — retained for provenance |
| `r6-profile-results-v2.json` | v2 | AUTHORITATIVE — all tables generated from this |
| `r6_4_*.json` | v1 | SUPERSEDED — footprint-based estimate |
| `r6_a_direct_*.json` | v2 | REPAIRED — direct pixel-level simulation |
| `r6_5_*.json` | v1 | SUPERSEDED — broken oracle (T_opt + traffic) |
| `r6_c_repair_*.json` | v2 | REPAIRED — bandwidth-limited traffic model |
| `r6_b_correctness_*.json` | v2 | NEW — correctness validation |

## No new approximate optimization started

This repair phase only corrected measurement and oracle errors. No new
optimization candidates were created. No approximate gradients, gradient
skipping, or loss/densification changes were introduced.

## GPU budget

- R6-A direct: 9 workloads × ~1 min = ~9 min (GPU 5)
- R6-C repair: 9 workloads × ~1 min = ~9 min (GPU 6)
- R6-B correctness: 3 workloads × ~1 min = ~3 min (GPU 7)
- Total: ~21 min GPU time
- No full training runs, no 13-scene expansion, no multi-seed

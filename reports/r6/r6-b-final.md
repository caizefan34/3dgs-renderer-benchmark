# R6-B — Final Gate

## Verdict

```
B0_SUCCESS  = FAIL  (no stable positive E2E benefit)
B1_SUCCESS  = FAIL  (systematic regression on all 9 workloads)
B1_DROP     = PASS  (touched-mask overhead exceeds zero-fill savings)
```

**R6-B = B0_NEUTRAL + B1_DROP**

B1-v2 is formally dropped. B0 is correct but provides no measurable E2E
benefit on A100 with gsplat 1.5.3. B0 may be retained as an infrastructure
option but does not meet the B0_SUCCESS gate (stable positive E2E benefit).

This does NOT invalidate the original observation that dense zero-init is
costly in principle — it only falsifies this implementation strategy on this
hardware/software stack. The zero-init cost on A100 is 0.03–0.13 ms, which
is negligible relative to the 20–38 ms backward.

---

## B0 — Persistent Buffer / Full Clear

### Correctness: PASS

- Rasterizer-only equivalence: bit-exact (max_abs=0.0 for all 5 gradients)
- Capacity growth: no stale data after buffer expansion
- Full-clear after invalidate: correct fallback

### E2E gain: NEUTRAL

- Mean Δiter: -0.11% (range: -0.39% to +0.03%)
- 7/9 workloads show slight regression, 2/9 show negligible gain
- The zero-fill cost (0.03–0.13 ms) is <0.2% of backward time on A100
- The allocation overhead that B0 eliminates is not a measurable bottleneck

### Verdict: B0_SUCCESS = FAIL

B0 is correct but does not provide a stable positive E2E benefit. The
persistent buffer infrastructure is sound and may be retained for future
experiments, but it does not meet the gate criterion.

---

## B1-v1 — Intersection-List Selective Clear (SUPERSEDED)

### Flaw confirmed

B1-v1 stored `prev_touched_ = flatten_ids [n_isects]` (intersection list with
duplicates) and launched one clear thread per intersection:

- O(n_isects) gradient writes — potentially MORE than B0's full clear
- Repeated clearing of the same row (correctness-safe, performance-invalid)
- Full intersection tensor retained (97 MB for room 15K vs 0.89 MB for B1-v2)

### Verdict: DO_NOT FORMALLY BENCHMARK

B1-v1 was never benchmarked. It is labeled SUPERSEDED.

---

## B1-v2 — Touched-Mask Selective Clear

### Correctness: PASS

**Sequential stale-gradient (rasterizer-only)**: PASS
- All stale rows (A-touched, B-untouched) are exactly zero
- B1-v2 at t+1 matches baseline bit-exactly

**Sequential stale-gradient (checkpoint, room 5K)**: PASS
- All 7 gradient types (mean2d, absgrad, xyz, SH, scaling, rotation, opacity)
  are exactly zero for A-touched/B-untouched rows
- B0/B1-v2 vs baseline: max_abs < 1e-6 (floating-point noise)

**Topology capacity growth**: PASS
- B1-v2 N=2 → invalidate → N=4: matches baseline within 5e-10
- No stale data from old buffer leaks into grown buffer

**Full-clear after invalidate**: PASS
- All stale rows exactly zero after invalidate-forced full clear

### Metadata overhead

| Component | Value |
|-----------|-------|
| Extra persistent bytes | C×N (bool mask) |
| Room 15K | 0.89 MB |
| Bicycle 15K | 3.77 MB |
| Garden 15K | 2.49 MB |
| Retained tensor lifetime | until next backward |
| Touched-mask preparation time | 0.13–0.27 ms (scatter kernel) |
| Selective-clear kernel time | 0.02–0.10 ms |

No hidden `unique`, sort, prefix-sum, or CPU synchronization.

### E2E gain: SYSTEMATIC REGRESSION

- Mean Δiter: -0.40% (range: -0.70% to -0.20%)
- **All 9 workloads show regression**
- The scatter phase (0.13–0.27 ms) costs 2–4× more than the full clear
  it's trying to eliminate (0.03–0.13 ms)
- The selective-clear savings are negligible because most Gaussians are
  touched in typical iterations (r_touch ≈ 0.6–0.8)

### Verdict: B1_DROP

The corrected touched-clearing overhead consumes the expected savings and
causes systematic regression. This is the B1_DROP condition.

This does NOT invalidate the original observation that dense zero-init is
costly. It only falsifies this implementation strategy: on A100 with gsplat
1.5.3, the zero-init is too fast (0.03–0.13 ms) for any selective-clear
approach to beat, because the selective-clear metadata preparation (scatter)
inherently costs more than the zero-fill itself.

---

## Gate summary

| Gate | Criterion | Result |
|------|-----------|--------|
| B0_SUCCESS | Stable positive E2E benefit + correctness PASS | FAIL (correctness PASS, no positive benefit) |
| B1_SUCCESS | Sequential PASS + topology PASS + no hidden preprocessing + stable E2E gain + no small-workload regression | FAIL (correctness PASS, systematic regression) |
| B1_DROP | Corrected touched clearing overhead consumes expected savings or causes regression | PASS (scatter 0.13–0.27 ms > clear savings 0.007–0.030 ms) |

---

## Report paths

- `reports/r6/r6-b-implementation-audit.md`
- `reports/r6/r6-b1-v2-design.md`
- `reports/r6/r6-b-correctness.md`
- `reports/r6/r6-b-benchmark.md`
- `reports/r6/r6-b-final.md` (this file)

## Commits

- B0: `14849f6`
- B1-v1: `b50afa7` (SUPERSEDED)
- B1-v2: `1be8f82` (+ `3329d4e`, `c7d05bc`, `8c8d8fb`, `d53f6d3`)

## B1-v1 history preserved

The original report `reports/r6/r6-b1-selective-zero.md` is retained.
B1-v1 source is preserved in git history at commit `b50afa7`.

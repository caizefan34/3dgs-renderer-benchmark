# P18-P0-R1 — Valid QSCAN Benchmark (FQ-GS oracle re-run)

**Verdict: `P18_P0_HARD_DROP` — FQ-GS is permanently closed (do not enter P18-P1).**
Date: 2026-09-28. GPU: GPU 3 (A100, GPU-431ac351), all 8 GPUs verified idle at each launch.
Code/artifact hashes: see `P18_R1_VERDICT.json`. Logs: `scripts/sass_audit_r3.log`, `scripts/eval_run_r2.log`.

## 1. Why R1 existed

The original P18-P0 result (QSCAN4/8/16 ≈ 0.966–0.972, `P18_P0_HARD_DROP`) was produced by an evaluator
that could not have shown speedup even if the idea worked. Its defects:

1. `exp2f(-dd*pos)` per lane in `r_pos` — the very "position-dependent exponential" the protocol forbids;
2. a per-lane `gd = direct_G(...)` recompute for the guard, plus
3. a guard fallback `if (|gq-gd| > GAMMA || branch differs) gq = gd` — i.e. DIRECT work in the QSCAN path;
4. wrong first-difference formula (`d0 = b + 2a·x0` instead of `a(2x0+1)+b`) on a synthetic `x = lane` layout;
5. the trace was Gaussian-footprint uniform sampling, not real contiguous subgroups.

R1 rebuilt everything: real-contiguous trace (R1-A/B), corrected evaluator (R1-C), mandatory SASS audit (R1-D),
gate re-run with the unchanged thresholds (R1-E).

## 2. What was rebuilt

- **Trace (R1-A/B)**: 16-px real contiguous row segments emitted by the frozen `macro_tile_rasterize_kernel`
  phase-2 inner loop; `x[j+1]-x[j] == 1` auto-asserted. Records: room 1,081,568 / bicycle 1,871,872
  (requirement ≥ 8192 — exceeded 132×/228×). Full-population histograms from the actual masks; record
  integrity re-verified bit-exact (0 mismatches of 2,952,440 records); neutrality FAIL interpreted as the
  base renderer's own per-pass bistable flake (documented in the manifest).
- **Evaluator (R1-C)**: allowed exps only — `G0=exp(-q0)`, `r0=exp(-d0)`, `k=exp(-2a)` on leader lanes;
  `k^j` by pure FMUL on the bits of `j`; exclusive multiplicative Hillis-Steele scan; no guard, no per-lane
  direct recompute (validation runs raw). Workload = one lane per (record, pixel t), record-major.
- **SASS audit gate (R1-D)** — `P18_R1_SASS_AUDIT.json`, PASS:
  - A: DIRECT keeps 1 MUFU/px; B: QSCAN4/8/16 all exactly 3 MUFU (constant in SUB, `MUFU.EX2` only);
  - C: dynamic exp count 1.0 / 0.75 / 0.375 / 0.1875 per px (DIRECT / Q4 / Q8 / Q16);
  - D/E: forbidden control (`exp(k·pos)` per lane) shows 4 MUFU statically and 1+3/SUB dynamically —
    the detection is sensitive; F: exp-free control certified 0 MUFU / 0 dynamic exps.

## 3. Gate re-run (unchanged rule: per-variant median S_eval over the 4 cells; ≥4.0 STRONG, <2.5 HARD_DROP)

| cell | S_eval q4 | q8 | q16 | S_eval noexp-bound |
|---|---|---|---|---|
| room / exp2 forward | 0.989 | 0.986 | 0.983 | 1.002 |
| room / exp backward | 0.988 | 0.985 | 0.983 | 1.002 |
| bicycle / exp2 forward | 0.988 | 0.985 | 0.983 | 1.002 |
| bicycle / exp backward | 0.988 | 0.985 | 0.983 | 1.002 |
| **median** | **0.9882** | **0.9853** | **0.9831** | **1.0017** |

Raw 1×-trace numbers agree to ±0.001. All three variants: **HARD_DROP** (all below 1.0, far from the 4.0 STRONG bar).

**The decisive measurement is the exp-free bound.** An audited kernel with identical loads/store and *zero*
exponentials runs at **1.0017×** the DIRECT time. The proxy is memory-bandwidth-bound (~85% of A100 HBM peak:
1.38 GB moved in 0.80 ms). The per-pixel exp costs ~28 µs of MUFU work against an 800 µs kernel — fully hidden.
Consequently **no exp-elimination technique of any kind can exceed ~1.0× in this proxy**; QSCAN's 0.98–0.99
medians are simply its scan/shuffle overhead on top of an unchanging memory roof.

## 4. Secondary finding: the legal QSCAN chain is numerically broken on real geometry

Raw no-guard validation (real trace, the whole population):

| scene / domain | NaN sub4 → sub16 | branch flips sub16 | max finite |err| |
|---|---|---|---|
| room / exp2 | 6,917 → 401,452 (2.3%) | 377,623 (2.2%) | 0.74 |
| room / exp0 | 29,588 → 634,216 (3.7%) | 619,713 (3.6%) | 0.73 |
| bicycle / exp2 | 30,300 → 1,307,539 (4.4%) | 1,243,543 (4.2%) | 0.57 |
| bicycle / exp0 | 110,489 → 2,025,012 (6.8%) | 1,983,708 (6.6%) | 0.88 |

(inf count 0; nearly all mismatches are far from the 1/255 threshold; from `P18_R1_EVAL_MICROBENCH.json`.)

Mechanism: a real 16-px row that clips the footprint edge has q spanning thousands; the leader anchor
`G0=exp(-q(xs))` underflows to 0 while `r0=exp(-d0)` overflows to inf within the same subgroup → `0·inf = NaN`;
FTZ-ed intermediate prefixes give finite-but-wrong values. Repairing this needs the per-lane direct value —
exactly the defect mode that invalidated P18-P0. The partially-live records (pop<16: 67%/79% of records) are
precisely the affected class.

## 5. Conclusion

The true, SASS-certified QSCAN delivers S_eval ≈ 1 **and** is numerically unusable on real geometry; and the
exp-free control proves the proxy itself has no headroom for any exp-elimination idea. Per spec item 7 this is
accepted as **`P18_P0_HARD_DROP`**; **FQ-GS is permanently closed**, no P18-P1.

Constraints honored: all original gates/thresholds/method unchanged; no git commit; no training; only GPU 3
touched; no auto-entry into any next phase.

## Provenance

- `P18_R1_SASS_AUDIT.json` (sha256 17bdc494…), `P18_R1_EVAL_MICROBENCH.json` (1fd16f5c…),
  `P18_R1_TRACE_MANIFEST.json` (d9d1a604…), traces `P18_R1_TRACE_room.npz` (54582d8e…),
  `P18_R1_TRACE_bicycle.npz` (a0f13b70…).
- Evaluator `p18_r1_eval.cu` sha1 adfbdc60…, scripts 41ab02b6… / 97c6239e… / 58725d29…,
  built `.so` sha256 8b15a79d…, frozen flags `-O3 -DNDEBUG -use_fast_math … sm_80`.
- Method identical to P18-P0 (`p18_p0_trace_eval.py time_eval`: CUDA graphs, 7 reps × 10 iters, per-replay
  median; `T_*_ms` keys carry true ms — the P18-P0 artifact's `_us` keys held ms-scale numbers, mislabeled).

# P18-P0 Final Closeout — FQ-GS Warp-Quotient Rasterization Oracle

**Final verdict: `P18_P0_HARD_DROP` — `FQ-GS CLOSED` — `NO P18-P1`.** STOP.

Date 2026-09-28. All timing on a single clean A100 (GPU 3, `GPU-431ac351-…`, A100-SXM4-80GB; all 8 GPUs verified idle at each launch). This is the **R1 closeout**: the final scientific verdict was frozen before this document, and this closeout only assembles the R1 evidence chain and records the two independent kill mechanisms.

---

## 1. The frozen conclusion

On the **frozen C0 rasterizer on the EPIC-05 A100**, under the real room/bicycle 16K workload, the warp-quotient exp-reconstruction (QSCAN4/8/16) is **permanently rejected**: it delivers no speedup **and** is numerically unusable on real geometry. Two independent mechanisms kill it (Section 4).

> **Scope — do not over-generalize.** This is a *frozen-C0 / A100 workload* conclusion. It is **not** a claim that exp-elimination is invalid on every GPU, every 3DGS backend, or every workload. The two kill mechanisms (memory-bound proxy with a hidden exp; leader-anchored under/overflow on footprint-edge rows) are specific to this backend + GPU + workload. A backend that fuses the exp into a memory op, a GPU with a different MUFU:HBM ratio, a half-precision exp2-only path, or a much higher live-lane-population workload is **out of scope** for this campaign.

---

## 2. Why R1 existed (pre-R1 invalidity)

The original P18-P0 run (QSCAN ≈ 0.966–0.972) was produced by an evaluator that could not have shown speedup even if the idea worked. Its defects (pre-R1 artifacts now marked `INVALID_SUPERSEDED_BY_R1` in `P18_EVIDENCE_CHAIN.json`):

1. a per-lane **position-dependent** exp in `r_pos` — the exact defect mode the protocol forbids;
2. a per-lane `gd = direct_G(...)` **recompute** for the guard, plus
3. a guard **fallback** `if (|gq-gd|>GAMMA || branch differs) gq = gd` — DIRECT work hidden in the QSCAN path;
4. the wrong first-difference formula (`d0 = b + 2a·x0` instead of `a(2x0+1)+b`) on a synthetic `x = lane` layout;
5. the trace was Gaussian-footprint uniform sampling, **not** real contiguous subgroups.

R1 rebuilt everything: **real contiguous trace** (R1-A/B), **corrected evaluator** (R1-C), **mandatory SASS audit gate** (R1-D), and the **gate re-run with unchanged thresholds** (R1-E).

---

## 3. Final performance numbers (S_eval, SASS-certified)

`S_eval = median T_DIRECT / median T_QSCAN` over the 4 cells `{room, bicycle} × {exp2 forward, exp backward}`, CUDA graphs + events, 7 reps × 10 iters, per-replay median, on the real contiguous trace.

| cell | S_eval Q4 | Q8 | Q16 | **exp-free bound** |
|---|---|---|---|---|
| room / exp2 forward | 0.989 | 0.986 | 0.983 | 1.002 |
| room / exp backward | 0.988 | 0.985 | 0.983 | 1.002 |
| bicycle / exp2 forward | 0.988 | 0.985 | 0.983 | 1.002 |
| bicycle / exp backward | 0.988 | 0.985 | 0.983 | 1.002 |
| **median over cells** | **≈ 0.9882** | **≈ 0.9853** | **≈ 0.9831** | **≈ 1.0017** |

Raw 1×-trace numbers agree to ±0.001. Every variant is at/below 1.0 (slightly slower than DIRECT) — far below the 4.0 STRONG bar and below the 2.5 hard-drop floor.

**The decisive measurement is the exp-free bound (≈ 1.0017).** A SASS-certified kernel with *identical loads/store and zero exponentials* runs at 1.0017× the DIRECT time. The proxy is **memory-bandwidth-bound (~85% of A100 HBM peak**: 1.38 GB moved in 0.80 ms); the per-pixel exp is ~28 µs of MUFU work against an 800 µs kernel and is **fully hidden**. Consequently **no exp-elimination technique of any kind can exceed ~1.0× in this proxy**; QSCAN's 0.98–0.99 medians are simply its scan/shuffle overhead on an unchanging memory roof.

**Implementation validity (SASS audit, PASS):** DIRECT = **1.0 exp/px**; QSCAN4/8/16 = **3/SUB per px (0.75 / 0.375 / 0.1875)** — strictly decreasing, confirming genuine exp elimination; QSCAN16 has **no position-dependent MUFU** (its only MUFU ops are the 3 leader exps G0/r0/k, `MUFU.EX2`; `k^j` is pure FMUL; the scan is SHFL+FMUL). The forbidden control (per-lane `exp(k·pos)`) shows 4 MUFU statically and 1+3/SUB dynamically, proving the audit detects per-lane exps. The exp-free control is certified 0 MUFU / 0 dynamic exps.

---

## 4. The two independent kill mechanisms

### A. Performance — no exp-elimination headroom (frozen C0/A100 proxy)
The exp-free, SASS-certified control (identical loads/store, 0 exps) times the **same** as baseline DIRECT (median 1.0017×). The inner loop is memory-bandwidth-bound and the per-pixel exp is fully hidden by memory latency. Eliminating the exp — which is exactly what QSCAN does (0.1875 exp/px for Q16, vs 1.0 for DIRECT) — buys **nothing** because the exp was not the bottleneck. QSCAN's measured slowdown (0.9831–0.9882) is the cost of its scan/shuffle work on top of a memory roof it cannot move.

### B. Numerical — leader-anchored recurrence is unusable on real footprint-edge rows
Raw no-guard validation on the real contiguous trace (the legal QSCAN chain: allowed exps only, pure-multiply `k^j`, multiplicative scan, no guard):

| scene / domain | NaN sub4 → sub16 | branch flips (sub16) | max finite \|err\| |
|---|---|---|---|
| room / exp2 forward | 6,917 → 401,452 (2.3%) | 377,623 (2.2%) | 0.74 |
| room / exp backward | 29,588 → 634,216 (3.7%) | 619,713 (3.6%) | 0.73 |
| bicycle / exp2 forward | 30,300 → 1,307,539 (4.4%) | 1,243,543 (4.2%) | 0.57 |
| bicycle / exp backward | 110,489 → 2,025,012 (6.8%) | 1,983,708 (6.6%) | 0.88 |

(inf count 0; nearly all mismatches are **far from** the 1/255 threshold.)

**Mechanism:** a real 16-px row that clips the footprint edge has `q` spanning thousands within the subgroup; the leader anchor `G0 = exp(-q(xs))` **underflows to 0** while `r0 = exp(-d0)` **overflows to inf** in the same subgroup → `0·inf = NaN`; FTZ-ed intermediate prefixes give **finite-but-wrong** values. Repairing this requires the per-lane direct value — **exactly the P18-P0 defect mode (per-lane recompute) that invalidated the original run.** The partially-live records (pop<16: 67% of room / 79% of bicycle records) are precisely the affected class; the fully-live records (pop16: 33%/21%) are the only favorable class, and even there there is no speedup (memory-bound).

The two mechanisms are independent: A kills the *performance premise* even where B is silent (fully-live rows), and B kills the *numerical soundness* on the majority (partially-live) rows. Either alone is disqualifying; both hold.

---

## 5. Decision and stop conditions

Per **spec item 7** (true QSCAN still S_eval ~ 1) and the **unchanged protocol-16 gate** (all variants median S_eval < 2.5), the verdict is **`P18_P0_HARD_DROP`**. **FQ-GS is permanently closed.** No P18-P1.

Constraints honored (no rescuing): all original gates/thresholds/method **unchanged** (4.0/2.5, γ, cells, method); no threshold/γ/tile/camera/quality change; no git commit; no full training; GPU 3 only (all 8 verified idle); no re-implementation of QSCAN beyond the SASS-audited R1 evaluator; no new experiments; no use of D0's main GPU.

---

## 6. Artifact index

**Final closeout (this task):**
- `P18_FINAL_CLOSEOUT.md` — this report
- `P18_EVIDENCE_CHAIN.json` — valid R1 evidence (R1-1…R1-7) + pre-R1 artifacts marked `INVALID_SUPERSEDED_BY_R1`
- `P18_VERDICT.json` — final verdict (supersedes the pre-R1 draft)
- `MANIFEST_P18_FINAL.json` — sha256 of all final + R1 + superseded artifacts

**Valid R1 evidence (`r1/`):**
- `P18_R1_TRACE_MANIFEST.json` + `traces/P18_R1_TRACE_{room,bicycle}.npz` — corrected contiguous trace (R1-1)
- `P18_R1_SASS_AUDIT.json` — SASS audit gate PASS; dynamic exp counts (R1-2, R1-3, R1-4)
- `P18_R1_EVAL_MICROBENCH.json` — exp-free bound + valid S_eval timings + raw no-guard failure (R1-5, R1-6, R1-7)
- `cuda/p18_r1_eval.cu`, `logs/sass_audit_r3.log`, `logs/eval_run_r2.log` — provenance

**Superseded (historical, `INVALID_SUPERSEDED_BY_R1`):** pre-R1 `P18_EVAL_MICROBENCH.json`, `P18_REFERENCE_VALIDATION.json`, `P18_TRACE_MANIFEST.json`, `traces/P18_TRACE_*.npz`, `P18_REPORT.md`, `P18_VARIANT_TABLE.json`, `MANIFEST_P18.json`.

**Frozen baseline (identity-verified):** trainer `675b0267…`, c0_ext `d26cccf4…`, core `935ea6c0…`, fused_ssim `dd6a4c79…`, ckpt room `17c1f590…`, ckpt bicycle `0c36f5c7…`.

---

*End of P18-P0. FQ-GS closed. No P18-P1. STOP.*

# Publication checkpoint 2 — P4 complete, Stage B underway, expansions queued

**Date:** 2026-09-24 (goal round 3) · **Supersedes:** checkpoint-1.md
**FINAL-30K result: UNCHANGED and preserved** (headline recheck from frozen artifacts: 1.0685×
geomean speedup, mean ΔPSNR +0.070, 13/13 faster — reproduced by the aggregation engine).

## Phase status

| Phase | Status | Evidence |
|---|---|---|
| P1 B1 gate | **3/3 clean runs**; room PASS, bicycle PASS, garden C3 flag under seed-43 resolution | `baselines/b1_*`, `b1_gate_verdict.json` |
| P1 B0 gate | 3/3 RUNNING (fixed trainer, past the crash point; ~3.7× slower than gsplat arms) | `log_b0_*.txt` |
| P1 13-scene expansions | **QUEUED** (b1 ×10, b0 ×10 after gates) | `pub_queue.json` (89 entries) |
| P2 Stage A | **COMPLETE 9/9 clean** (room/bicycle/garden × a0/a1/a2), artifacts local | `ablations/stage_a/` |
| P2 Stage B | **RUNNING** (30 runs queued, a0_bonsai first) | scheduler |
| P4 eps2d 2×2 | **COMPLETE 9/9 clean** → **MATERIAL_CONFOUND (quality axis)** | `eps2d/p4_eps2d_verdict.json` |
| P6 multiseed | **QUEUED** (16 runs: b1a/c0 × seeds 43/44 × 4 scenes) + garden s43 pair in flight | queue |
| P5 external | not started (after P1/P2 underway — condition now met; Faster-GS next) | — |
| Aggregation pipeline | **BUILT + SELF-VALIDATED** (headline check reproduces 1.0685×/+0.070 exactly) | `aggregates/runs_master.json` |
| Figures/tables | A–G + 1–4 generated from frozen artifacts; 5–6 pending P6/P5 | `figures/`, `tables/` |

## P4 verdict (the round's main result)

Pre-registered rule fires on **quality**: C0's eps2d effect −0.100 dB > FINAL-30K mean 0.070.
Wall does not fire (1.07% < 3.42%). Direction analysis:
- eps2d=0.3 makes C0 **1.07% faster** → headline would be ~1.057× at matched eps2d.
- eps2d=0.3 makes C0 **0.10 dB lower** → headline ΔPSNR would be ~+0.170 at matched.
- Biases oppose; "faster at quality-neutral-to-positive" is robust under both values.
- **Every C0-vs-B1A table must carry the disclosure** (see eps2d-sensitivity.md §Reporting).

## B1 gate — garden question: **RESOLVED as a systematic accutile effect (NOT noise)**

- room +0.187 / bicycle −0.030 / garden **−0.642 dB** (C3 band ±0.50).
- Trajectory analysis showed late-training divergence; the working hypothesis was
  trajectory noise — **REFUTED by the seed-43 pair**:
  - b1a garden: s42 24.194 / s43 24.969 · b1 garden: s42 23.552 / s43 24.163
  - b1−b1a delta: **s42 −0.642, s43 −0.806 dB** — reproduced at the second seed, larger.
  - Seed-to-seed within-arm spread (+0.61/+0.78) does NOT explain the between-arm delta.
- **Finding: accutile (exact tile accumulation) systematically improves garden final
  PSNR by ~0.6–0.8 dB** vs pristine gsplat, on top of its consistent +2.2–4.6% wall
  edge (all scenes, both seeds). B1A is not merely "B1 + speed".
- Verdict: **B1_REPRODUCTION_PASS_WITH_FINDING** (`b1_gate_verdict_v2.json`) — wiring
  verified (C1/C2/C4/C5/C6 pass everywhere); the C3 exceedance is real and attributed.
- Headline UNAFFECTED: C0-vs-B1A both use accutile → the effect cancels in the matched
  comparison (garden ΔPSNR −0.032). C0-vs-B1 quality columns carry the effect and must
  be labeled accordingly (or avoided in favor of B1A-based comparison).
- Seed-44 garden pair queued for 3-seed confirmation (b1as44/b1s44).

## Stage A first ablation numbers (3 scenes, clean)

| increment | speedup geomean | mean ΔPSNR | note |
|---|---|---|---|
| A0→A1 (F9) | 0.996 | +0.166 | wall-neutral; garden +0.581 |
| A1→A2 (SCALAR_ADJOINT) | 1.002 | −0.128 | wall-neutral; garden −0.626 |
| A2→A3=C0 (H8-MR) | 1.008 | +0.012 | small consistent wall gain, 2/3 faster |
| A0→C0 (full stack) | 1.006 | +0.049 | |
| B1A→B1 (accutile off) | 0.958 | −0.162 | accutile = +4.4% baseline speed |
| B1→C0 | 1.115 | +0.194 | eps2d-asymmetric; P4 bounds it |

Per-module wall deltas are small and not individually attributable (§9 discipline: no
causal kernel attribution from wall time). Garden wiggles ±0.6 dB between EVERY adjacent
ablation pair — per-scene quality deltas at seed 42 are noisy at that scale on garden.

## B0 gate observations (room + garden done, bicycle in flight)

- **B0 matched-protocol numbers (PUBLICATION grade):** room 47.7 min / PSNR 31.695 /
  N 755K; garden 45.7 min / 24.391 / 2.00M. vs B1A: **1.94–2.56× slower**, ΔPSNR
  −0.239/+0.197 (same band), **N −12% to −22%** (the original signed-grad-norm
  densification statistic is stricter → fewer Gaussians).
- The two B0 integration bugs (opacity [N,1]; radii 1-D → 0-D mask cascade) were fixed
  and verified by a 200-iter trainer smoke (RC=0); the full runs densified normally
  (N 528K → 755K room) and completed clean.

## P5 external systems — data discovered COMPLETE

All three external systems already have **13 scenes × {native, c42} = 26/26 runs each,
all rc=0** (earlier strong-baselines phase; `strong_baseline_results/*/all_metrics.json`):
- **faster-gs** @3cb0b75 (clean worktree)
- **fastgs** @d4d33b6 (dirty = __pycache__ only; commit pinning sound)
- **speedy-splat** @b9dd42d (dirty = submodule pointers; record submodule commits)
System classification: all three are SYSTEMS (custom densification/optimizer + renderer),
→ SYSTEM_LEVEL_COMPARISON (Table 6), not matched-protocol. `p5_external.json` +
`table6_p5_external.md` generated. Scene-dependent failures disclosed (e.g. Faster-GS
garden 13.4 dB under its own config). Provenance caveat: wall times from the earlier
phase's scheduling — disclosed as SYSTEM_LEVEL timing.

## Infrastructure (this round)

- Scheduler: fixed adopted-run reaping (rc=None judged by results.json); file-driven queue
  (`pub_queue.json`, re-read each cycle → extensions without restart); state patches only
  while dead.
- Aggregation engine `aggregate_publication.py`: scans final30k_runs + pub_runs →
  `runs_master.json` (43 runs) + pair tables; §19-consistent orientation validated against
  the FINAL-30K headline.
- `make_figtables.py`: figures A–G + tables 1–4 from the aggregates (regenerable as runs land).

## Run matrix (throughput ledger)

- done: 19 publication runs (Stage A 9, B1 gate 3, P4 corners 6, c0e01_room clean, b1as43_garden)
- in flight: 5 (b0 gate ×3, b1s43_garden, a0_bonsai)
- queued: 65 (Stage B 29, b1 ×10, b0 ×10, P6 16)
- contamination: 1 historical attempt preserved (`c0e01_room_contaminated_attempt1`)

## Next

1. b1s43_garden lands → B1_REPRODUCTION final verdict (expect PASS_WITH_DISCLOSURE).
2. b0 gate ×3 lands → B0 functional + timing verification vs the 3.7× preliminary read.
3. Stage B grinds (≈2 h at 5 GPUs); then b1/b0 expansions + P6.
4. Faster-GS 3-scene gate (P5) once ≥3 GPUs free after Stage B.
5. Tables 5–6 + remaining reports as their data lands; regenerate figtables on each pull.

## Blockers

None. (Password expiry warning ~5 days — renewal needed for continued mx access.)

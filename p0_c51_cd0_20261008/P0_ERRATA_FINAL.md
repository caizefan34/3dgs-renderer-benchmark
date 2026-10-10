# P0 ERRATA (FINAL) — C51-CD0-MX Stage 0

Status: `P0_NO_NOVELTY` **UPHELD** (scope narrowed — see E4). Originals in this dir are NOT overwritten;
this file is the authoritative correction record. Read together with `P0_REPORT.md` / `P0_VERDICT.json`.

## E1 — Terminology: "matched population" -> "densification-signal-preserving control"
- Where: `P0_REPORT.md` §4 H1, §3; `P1_PREREG.md`; `P0_VERDICT.json` hypotheses.
- Correction: replace "matched population" with **"densification-signal-preserving control"**.
  Condition C (Stage5 `compute_densify_grad=True`) preserves the *densification signal* but does NOT
  hold Gaussian population identical to baseline. Measured C/A final-Gaussian ratios (Stage5 archival):
    - Room    C/A = 0.946  (A 2,155,755 -> C 2,039,821)
    - Bicycle C/A = 1.119  (A 11,049,676 -> C 12,359,659, over-densifies)
    - Garden  C/A = 0.997  (A 11,401,827 -> C 11,365,479)
- Consequence: H1's falsification rests on C-vs-A *iter-time* (Room +0.5%, Bicycle -6.6%, Garden -1.7%)
  and on the C-vs-D semantic-equivalence test, NOT on population being held equal. The population drift
  above is an additional, quantified mismatch, not a claim of exact population matching.

## E2 — "94% from densification" is a Room-Stage-5 attribution approximation only
- Where: `P0_REPORT.md` §2, §3, §4 closing answers; `P0_VERDICT.json` `archival_attribution`.
- Correction: state as "an **experimental attribution approximation for Room Stage 5** (B-C = +7.7pp of
  the +8.2% total), NOT a strict causal population ratio." The Stage5 report itself flags that B and C have
  different Gaussian populations, optimizer states, and stochastic trajectories; B-C cannot be unconditionally
  attributed to one module or generalized across scenes. Bicycle/Garden C/A > 1 (over-densification) further
  warn against extrapolating the Room 94% figure. No cross-scene causal share is claimed.

## E3 — LAW-OPT-FRAC scope is optimizer-side only
- Where: `NOVELTY_DELTA_MATRIX.md` row 4; `P0_REPORT.md` §4 H2.
- Correction: LAW-OPT-FRAC bounds **optimizer-side** overhead (optimizer = 5.94-14% of wall on Faster-GS fused;
  even all-BF16 caps E2E at 1.017-1.042x). It does **not**, by itself, prove infeasible *all* future
  raster-cost-control mechanisms. Forward/raster-cost levers are bounded by separate results (LAW-AABB-TILE,
  LAW-BEVT, TFC certified footprint work). C51-CD0's NO_NOVELTY does not rest on LAW-OPT-FRAC alone; it rests
  on the coupling mapping to FastGS (capacity) + VTS0/ReBudgetGS (topology/capacity scheduling) + the falsified
  backward lever. LAW-OPT-FRAC is supporting, not decisive, and its scope is explicitly optimizer-side.

## E4 — P0_NO_NOVELTY scope limited to THIS C51-CD0 design
- Correction: the verdict applies to the **specific C51-CD0 "compute-densification co-design" hypothesis** as
  posed (joint gradient-spend + capacity-growth coupling). It is NOT a verdict on the entire 3DGS-training
  acceleration research direction. Other axes remain open and are the stated GPU priority: **TC-R1
  (Temporal Coverage)** and **TFC (Tight Footprint Culling)**. Do not read this as "3DGS training acceleration
  is a dead end."

## E5 — SOURCE_IDENTITY tracked RUN_STATUS.json status corrected
- Where: `SOURCE_IDENTITY.json` `tracked_changes`.
- Correction/clarification: the tracked modified file is
  `WORK_GRADIENT_ORACLE0_MX/WORK_GRADIENT_ORACLE0_MX/RUN_STATUS.json` (git status ` M`, unstaged).
  This modification is **pre-existing / external to Stage 0** — it was already dirty at session start and is
  NOT a product of this read-only audit. Stage 0 itself wrote only inside `p0_c51_cd0_20261008/` (untracked)
  and modified no tracked project files. The two submodule gitlinks (`public_baselines/gsplat`,
  `clockgs/third_party/dgr_clock`) are likewise pre-existing `-dirty` gitlinks, unrelated to this stage.

## E6 — Unified abbreviations
- **TC-R1 = Temporal Coverage** (first-priority authorized GPU track after C51-CD0 closure).
- **TFC = Tight Footprint Culling** (the certified forward-pipeline footprint-tightening lever; PAPER0
  MAIN_CANDIDATE). TFC is a forward-pipeline method and is distinct from TC-R1.
- Correction applied wherever the prior text wrote "Temporal Coverage (TFC)" as if one thing: they are two
  separate items. Closing GPU priority order: **TC-R1 first, then TFC strong-baseline end-to-end + originality review.**

## E7 — Archival + closure statement
- Original P0 evidence (`P0_REPORT.md`, `P0_VERDICT.json`, `NOVELTY_DELTA_MATRIX.md`, `P1_PREREG.md`,
  `LEGACY_RECOMPUTE.json`, `SOURCE_IDENTITY.json`, snapshots) is preserved unchanged.
- C51-CD0 is CLOSED: **no P1, no 30K rerun, no search for isomorphic methods.**
- GPU resource priority: (1) TC-R1 existing 10K aggregation verdict; (2) TFC strong-baseline E2E + independent-origination review.

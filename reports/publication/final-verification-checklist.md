# Final verification checklist (publication phase close-out) — EXECUTED 2026-09-24

Executed once the queue is empty (91/91 done, 0 failed) and P6 is analyzed. Every item
is mechanical; every number regenerates from frozen artifacts. No item may be checked
by narration — each requires a command or a file read.

**Execution record: `scripts/publication/verify_final.py` (A+B, remote),
`verify_f30k_untouched.py` (E1, remote), `verify_traceability.py` (E3, local);
C/D via the quoted greps. Result: ALL CHECKS PASS.**

## A. Run completeness — PASS (23/23)

- [x] `sched_state.py`: DONE == 91, failed == 0, ACTIVE empty; CONTAMINATED reviewed —
      three incidents (b0_counter, c0e01_room earlier, b1as43_train final), each with
      the renamed `_contaminated_attempt1` dir + `.snapshots/` forensics kept, each
      re-run clean. None silently dropped.
- [x] `queue_audit.py`: planned=91, state(done+active)=91, queue-entries=91 —
      MISSING none, EXTRA none.
- [x] Local evidence trees: 117 canonical remote run dirs verified present with
      `results.json` (26 FINAL-30K + 39 a-runs + 13 b1 + 13 b0 + 6 eps2d corners +
      16 P6 + 4 garden seed pairs); local `artifacts/publication/pulls/` holds the
      tar pulls + per-run extracts.
- [x] `finalize_stage_b.py` final output archived (39+13 verified, ladder COMPLETE,
      headline UNCHANGED-OK).

## B. Regeneration (all derived, nothing hand-transcribed) — PASS

- [x] `aggregate_publication.py` rc=0 → `aggregates/*.json` (runs_master + p1/p2/p4/p5
      + headline + garden_seed3; 117 runs in master).
- [x] `make_figtables.py` rc=0 → `figures/` A–G (6 PNGs, 43–190 KB each, no
      placeholder) + `tables/` 1–6.
- [x] Self-check line: headline geomean 1.0685, mean ΔPSNR +0.070 — UNCHANGED-OK at
      every regeneration (B3 asserts the values).
- [x] `p6_analysis.py` rc=0 → `p6_results.json` (per-seed geomeans
      1.0705/1.0735/1.0561; drjohnson REPRODUCIBLE). Contamination guard added: the
      script now refuses to emit timing when any P6 run carries the sticky flag.
- [x] All figures/tables re-pulled to local tree and verified non-empty (final pull
      139 files + corrected-artifact re-pull).

## C. Table-level discipline — PASS

- [x] §19: every speedup table is geomean-of-per-scene-ratios; every reduction figure
      separately labeled "reduction"; grep for "6.85" → only "speedup excess" usages
      plus the N-02 rule definitions. Table 1 ratio columns renamed to literal wall
      ratios with orientation note.
- [x] §20: "quality-neutral within the matched benchmark" present in the 5 key
      documents (claim map, summary, ablation, outline, checklist); grep for
      "no quality loss | quality-improving | lossless quality" → only forbidden-list
      definitions, zero usages.
- [x] eps2d disclosure travels with every C0/B1A table (C-13 bound 1.057×–1.0685×,
      quality −0.100 dB against C0).
- [x] drjohnson appears with its per-scene label in Table 1, Table 5, the claim map
      §4.3, and every text citing the mean ΔPSNR.
- [x] Garden-data caveat present in the external-systems garden mentions (C-15:
      SUPPORTED_WITH_CAVEAT).
- [x] Table 6 (external) carries SYSTEM_LEVEL + not-matched-protocol labels; no
      C0-vs-external speedup anywhere (N-05).

## D. Report-level closure — PASS

- [x] Every report status line is final; grep for "queued | in flight | pending |
      INTERIM | IN PROGRESS" → remaining hits are only the historical record
      (checkpoint-1/2.md, the interim-vs-final comparison sentence in
      cumulative-ablation.md) and this checklist's own spec.
- [x] `publication-summary.md` status: **COMPLETE** (92 runs: 91 planned + 1 clean
      contamination re-run, 0 failed); results section includes the P6 numbers.
- [x] `multi-seed-confirmation.md`: P6 tables filled from `p6_results.json`; R-13
      verdict REPRODUCIBLE recorded.
- [x] `matched-baselines.md`: 13-scene B1/B0 sections final; N-06 marked RESOLVED.
- [x] `claim-evidence-map.md`: C-03 caveat narrowed (13-scene single-seed; 3-seed on
      garden pairs + P6 subset), E22 VALID, R-09/R-13 rows closed.
- [x] `evidence-inventory.md` §6: E22 VALID with the final numbers.
- [x] `paper-evidence-outline.md` §8.5/§8.6: multi-seed lines final.
- [x] `reviewer-risk-audit.md` + `publication-readiness.md`: R-09 RESOLVED, R-11
      RESOLVED, R-12 RESOLVED, R-13 RESOLVED-REPRODUCIBLE, R-14 RESOLVED (3-seed),
      R-19 updated to 3-seed evidence; MULTI_SEED_ROBUSTNESS RESOLVED, ABLATION
      RESOLVED-COMPLETE, MATCHED_BASELINES STRENGTHENED-COMPLETE, TABLES COMPLETE.
- [x] `artifacts/publication/README.md`: index lists every final subtree + the
      regeneration commands (paths verified).

## E. Cross-checks — PASS

- [x] FINAL-30K untouched: publication phase started 2026-09-24 02:26:57 (first
      scheduler log line); ZERO files under `final30k_runs/` newer than that moment;
      26 canonical run dirs intact (+1 `_contaminated_attempt1` forensics dir from the
      cohort's own documented b1a_counter incident, predating the phase). Headline
      values reproduce exactly: 1.0685 / +0.070.
- [x] The three headline-restating documents (summary, claim map C-01, README) all
      quote 1.0685× / 13-of-13 / +0.070 dB identically.
- [x] No report cites a number absent from a frozen aggregate JSON — 7 most-cited
      verified programmatically: headline (1.0685/+0.070), P2 ladder (1.0045, 10/13,
      +0.079), B1 geomean (0.9662), B0 geomean (0.4142), P6 geomeans
      (1.0705/1.0735/1.0561), garden 3-seed (+0.642/+0.806/+1.132), R-13 trajectory
      (peak +3.369 @15000, recovery +4.741, per-seed deltas +1.102/+0.787/+1.597 —
      the trajectory fields were added to `r13_drjohnson_review.json` so the
      +3.37/+4.74 citations in the reports are JSON-backed).

**Verdict: publication-phase evidence set CLOSED. The FINAL-30K headline
(1.0685× geomean speedup, 13/13 scenes faster, mean ΔPSNR +0.070 dB,
"quality-neutral within the matched benchmark") stands unchanged, now with
matched baselines, a full cumulative ablation, eps2d bounds, 3-seed garden +
4-scene × 3-seed subset confirmation, and system-level external context.**

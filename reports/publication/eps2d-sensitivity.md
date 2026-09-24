# P4: eps2d sensitivity (2×2) — room / bicycle / garden, seed 42, 30K

**Status: COMPLETE.** All 9 cells run clean (PUBLICATION timing grade, no contamination).
Frozen evidence: `artifacts/publication/eps2d/` (run trees), `p4_eps2d_verdict.json`,
`tables/table4_p4_eps2d.md`, `figures/figE_eps2d.png`.

## Question

The headline comparison trains C0 at eps2d=0.3 (its frozen production value) against B1A
at eps2d=0.1 (gsplat's default). The eps2d axis therefore differs across the two arms of
the headline. P4 quantifies whether this asymmetry materially confounds the headline, per
the pre-registered rule:

- **EPS2D_NEGLIGIBLE** — geomean wall effect ≤1% AND |ΔPSNR| ≤ 0.10 dB on ≥2/3 scenes, both stacks.
- **MATERIAL_CONFOUND** — C0 wall effect > half the headline speedup (3.42%), or C0's mean
  |ΔPSNR| exceeds the FINAL-30K mean |ΔPSNR| (0.070 dB).
- **SMALL_BUT_PRESENT** — otherwise.

## The 2×2 (seed 42, clean pairs)

| scene | stack | wall@0.1 (min) | wall@0.3 (min) | wall ratio | ΔPSNR (0.3−0.1) | ΔSSIM | ΔLPIPS |
|---|---|---|---|---|---|---|---|
| room | B1A | 18.7 | 18.6 | 0.9969 | +0.279 | +0.0014 | −0.0034 |
| room | C0 | 17.4 | 17.3 | 0.9970 | −0.144 | −0.0004 | +0.0014 |
| bicycle | B1A | 26.0 | 25.5 | 0.9827 | −0.119 | −0.0017 | +0.0008 |
| bicycle | C0 | 24.7 | 24.2 | 0.9808 | −0.053 | −0.0006 | −0.0002 |
| garden | B1A | 23.6 | 24.0 | 1.0184 | +0.012 | −0.0024 | +0.0078 |
| garden | C0 | 22.6 | 22.4 | 0.9903 | −0.104 | −0.0039 | −0.0003 |

Stack aggregates: **B1A** geomean wall ratio 0.9992 (0.08% effect), mean ΔPSNR +0.057.
**C0** geomean wall ratio 0.9893 (**1.07% effect** — 0.3 is slightly faster), mean ΔPSNR
**−0.100** (0.3 is slightly lower quality).

## Verdict: MATERIAL_CONFOUND (quality axis) — with direction analysis

The pre-registered rule fires on the quality axis: C0's mean |ΔPSNR(0.3−0.1)| = 0.100 dB
exceeds the FINAL-30K mean |ΔPSNR| = 0.070 dB. The wall axis does NOT fire (1.07% < 3.42%).
So the eps2d asymmetry is **material and must be disclosed in the headline tables**, not
silently absorbed. The direction analysis shows what the disclosure means:

1. **Speed bias favors C0.** C0@0.3 trains ~1.07% faster than C0@0.1. At matched
   eps2d=0.1 the headline speedup would be **~1.057×** instead of 1.0685×.
2. **Quality bias disfavors C0.** C0@0.3 is 0.100 dB lower than C0@0.1 on average. At
   matched eps2d=0.1 the headline mean ΔPSNR would be **~+0.170 dB** instead of +0.070.
3. **The two biases point in opposite directions and do not flip the conclusion.**
   Matched-eps2d yields a slightly smaller speedup with slightly better C0 quality —
   "faster at quality-neutral-to-positive" holds under both axes values.

## Reporting requirements (applies to every table carrying C0 and B1A)

- State C0's eps2d=0.3 and B1A's eps2d=0.1 explicitly in every such table.
- State the bounded effect: "the eps2d asymmetry contributes ~1.1% of the headline
  speedup and biases the quality comparison against C0 by ~0.10 dB; see P4".
- Never claim the comparison is eps2d-matched. It is not; it is the frozen production
  configuration of each system, with the asymmetry quantified.

## Per-scene notes

- garden ΔPSNR effects are within garden's demonstrated seed/trajectory noise (the same
  scene shows ±0.6 dB across A0/A1/A2 and ±0.8 dB across seeds 42/43), so its −0.104 is
  not individually reliable; the 3-scene mean is the pre-registered unit.
- The B1A stack is insensitive (0.08% wall) — accutile's exact-tile accumulation does not
  interact with the eps2d dilation on these scenes.
- eps2d does not act as a sparsity-control lever here: N ratios move ≤ ±9% with no
  consistent sign.

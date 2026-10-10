# P1_PREREG — Compute-Densification Co-Design

## Decision: NOT AUTHORIZED (P0 = NO_NOVELTY)

No P1 GPU experiment is pre-registered because Stage 0 found no non-trivial, non-isomorphic
mechanism. The coupling hypothesis collapses into: skip-backward (FALSIFIED lever, Stage5 C-vs-A
<5% all scenes) + density control (FastGS prior art) + topology/capacity scheduling
(CLOSED: VTS0, ReBudgetGS, LAW-OPT-FRAC). Pre-registering GPU work here would violate the
research protocol (no optimization before a surviving mechanism + profile).

## The single falsification that COULD reopen a lane (NOT launched)
IF a future claim is that a *backward/coupling* lever (not forward/TFC) can beat the strong
baseline at matched quality, the minimal counterfactual is exactly the archival Stage5 A/B/C/D
design — but on the CURRENT strong baseline (Faster-GS fused, A100/EPIC-05), NOT on the old
gsplat-1.5.3 + original-CUDA cohort:
  A = strong-baseline full densification, full backward
  B = candidate (coupled grad-spend + birth budget)
  C = candidate grad-spend with densification preserved (isolates optimizer/coupling effect)
  D = strong-baseline backward + masked optimizer (isolates compute-skip vs discard)
Required to even be admissible: matched init/seed/camera order/loss/optimizer; per-substage
timing (fwd / bwd / optimizer / topology); wall + PSNR/SSIM/LPIPS; >=2 scenes x >=2 seeds;
gate >=5% time-to-quality with dPSNR>=-0.10, dSSIM>=-0.005, dLPIPS<=+0.005 AND independent
contribution vs the strong baseline. Given Stage5 already shows C-vs-A <5% on all scenes and
LAW-OPT-FRAC caps optimizer savings <1.05x, the a-priori probability of passing is very low.
Do NOT launch without new independent evidence overturning a CLOSED law.

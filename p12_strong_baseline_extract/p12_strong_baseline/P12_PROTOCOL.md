# P12 Strong-Baseline Revalidation — Research Rationale

P11 established that the frozen trainer's separable PyTorch SSIM path is much
slower than the existing P11-validated fused-SSIM implementation while matching
the frozen loss semantics within calibrated floating-point tolerance.

Therefore, the next scientifically useful action is not another optimization
search. It is to revalidate the paper's two performance-sensitive conclusions
under the stronger common loss baseline:

1. current gsplat vs C0 complete-system endpoint;
2. A0 vs A1 (F9) original full-training effect.

H8 is explicitly excluded from rescue testing because two independent A100
systems already show full-training parity.

## Why P12-A comes first

The system-endpoint result is the paper's main modern-competitiveness context.
Removing a large common loss cost may change the observed end-to-end ratio even
if the renderer execution gap is unchanged. P12-A therefore updates the headline
system comparison.

## Why P12-B is still necessary

F9 is the only original named transform with a previously stable full-training
effect. A stronger common loss baseline can reduce dilution and make its actual
training contribution easier to measure, but it can also reveal that the old
~1% result was within system-level noise. Only a fresh paired 13-scene campaign
can resolve that.

## Why publication campaigns should not overlap on one host

F9 is a small effect. Even with separate GPUs, host-side CPU, I/O and scheduling
contention can be comparable to ~1%. P12-A and P12-B may be prepared in
parallel but their publication 30K windows should be non-overlapping on EPIC-05.

## Stop rule

After P12, no further exact-optimization search is authorized unless the paper
strategy itself changes. The result should be accepted whether it strengthens,
weakens or removes the existing performance claims.

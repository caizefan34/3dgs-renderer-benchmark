# 3DGS storage compression protocol

## Research question

Which representation minimizes stored bytes for a trained 3DGS checkpoint
subject to a declared round-trip quality and deployment-cost constraint?

"Most lossless" is not a valid technical category. Use the following terms.

| Class | Required property | Ranking boundary |
| --- | --- | --- |
| Bit-exact lossless | Decoded canonical tensors are byte- or value-identical under a declared canonicalization | Rank by size and encode/decode cost only |
| Same-checkpoint near-lossless | No retraining; decoded asset passes numerical and visual gates | Rank by a constrained size/deployment Pareto frontier |
| Retraining-required compression | Encoder or representation changes model optimization | Separate cohort with training cost and final quality |
| One-way delivery format | No canonical round-trip decoder | Report deployment utility; exclude from round-trip claims |

## Frozen evaluation contract

For every codec and scene, record:

- source checkpoint and canonical tensor hashes;
- codec source, commit/version, parameters, and deterministic seed;
- encoded bytes and compression ratio against the same canonical reference;
- encode and decode wall time, peak host memory, peak GPU memory, and upload time;
- decoded tensor error by parameter group;
- PSNR, SSIM, LPIPS, visual audit, and temporal consistency on fixed cameras;
- post-decode renderer identity and latency;
- failure, unsupported, and non-round-trip states without imputation.

The existing strict same-checkpoint gate is PSNR drop below 0.2 dB, SSIM drop
below 0.002, LPIPS increase below 0.005, plus visual review. A result that passes
this gate is near-lossless under this protocol, not mathematically lossless.

## Current supported result

On the frozen five-scene A100 cohort, SPZ 8/8 passes every declared near-lossless
gate at 5.572x to 6.072x compression and under 0.02 dB absolute PSNR change.
XZ is the bit-exact option with much smaller storage savings. These statements
do not establish a universal best codec.

## Submission blockers

1. Add encode/decode latency, peak decode memory, and render-ready upload cost.
2. Evaluate official learned codecs such as FCGS and current HAC-family methods
   in a separate retraining-required cohort.
3. Expand datasets and parameter distributions; include very large scenes and
   low-order/high-order SH cases.
4. Report rate-distortion and deployment-cost Pareto curves with uncertainty.
5. Obtain an independent visual audit and publish the blinded decision record.

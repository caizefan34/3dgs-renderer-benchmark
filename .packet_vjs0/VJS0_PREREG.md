# VJS0 — Visibility/Jacobian Superstep Feasibility Oracle (FROZEN)

## 0. Question

After ONE exact C0 render/backward linearization at theta0, can a frozen local image-space operator
support >=4 useful cheap inner parameter updates before an exact renderer refresh is required?

This is an upper-bound/mechanism probe, not a publishable method and not a 3DGS-LM replacement.

## 1. Cells

room/bicycle @10K/15K.

Per cell:
- 8 deterministic midpoint-stratified TRAIN source cameras.
- 8 corrected true-holdout cameras for evaluation only.
- topology/densification frozen throughout each probe.

## 2. Exact local operator

At theta0 for one source camera, build a local first-order image model:

    I(theta0 + delta) ~= I0 + J delta

The implementation may use exact autograd JVP/VJP or a custom cached-Jacobian representation, but
MUST pass parity:
- 32 frozen random optimizer-unit directions delta with ||delta|| matched to 0.25x native Adam step.
- compare J delta to finite-difference exact-render RGB response.
- median relative response error <= 1e-3
- p95 <= 5e-3
Otherwise: VJS0_LINEARIZATION_IMPLEMENTATION_FAIL.

## 3. Inner solver

Use the SAME C0 loss:
    0.8*L1 + 0.2*(1-SSIM)

Frozen local model only; no exact neighbor re-render during inner updates.

For K in {2,4,8}:
1. compute local-model loss/gradient at current delta;
2. apply a native-Adam-form update in optimizer-unit coordinates using a CLONED optimizer state;
3. update delta;
4. DO NOT refresh I0/J.

Important:
- the inner path is an optimistic oracle; cost is measured for local-model apply + optimizer only.
- final candidate theta_K is evaluated by the exact C0 renderer.
- optimizer state from the candidate inner path is preserved for evaluation but no densification.

Reference path:
- K exact native Adam steps on the SAME source camera, refreshing the exact renderer every step.

## 4. Trust diagnostics

At each inner step record:
- exact RGB linearization error after exact post-hoc render
- source exact loss
- local predicted loss
- parameter displacement norm
- Hellinger distance (diagnostic only; no gate tuning)
- tile/order churn (diagnostic only; ET0 already closed it as incremental trust signal)

No adaptive early stop. K is fixed.

## 5. Metrics

Per K:
- exact source progress capture:
      (L0 - L_candidate_K)/(L0 - L_exact_K)
  only when exact denominator >0; report validity count.
- exact peer-train mean-loss delta candidate vs exact-K
- true-holdout PSNR candidate vs exact-K
- local-model cost / exact-render-backward cost
- amortized predicted speed:
      K * T_exact / (T_build_J + K*T_inner)
- exact refresh equivalent:
      number of inner updates per exact build
- final RGB linearization relative error

## 6. Mechanical verdict

Primary K=4. K=2 and K=8 diagnostics only.
A SINGLE K=4 policy must pass ALL 4 cells.

### VJS0_STRONG_PROMOTE
- valid source-progress denominators >=7/8
- source progress capture median >=0.90
- p10 >=0.75
- true-holdout PSNR delta median >=-0.02 dB
- worst >=-0.08 dB
- final RGB linearization error median <=0.03
- amortized predicted speed >=3.0x
- local inner cost <=0.15x exact step
- no NaN/Inf

### VJS0_PROMOTE
- valid denominators >=6/8
- progress capture median >=0.80
- p10 >=0.60
- true-holdout median >=-0.04 dB
- worst >=-0.12 dB
- final RGB error median <=0.05
- predicted speed >=2.5x
- inner cost <=0.20x exact step
- no NaN/Inf

Else VJS0_DROP.

## 7. Interpretation

PROMOTE only says a frozen exact local renderer operator may remain useful for several updates.
It does NOT establish novelty vs 3DGS-LM or a real >=3x system.

Any follow-up MUST compare directly against 3DGS-LM / 3DGS^2-TR and must not reuse ET0/S0 rescue
arguments.

## 8. Forbidden rescue

No:
- adaptive K
- K outside {2,4,8}
- line search
- trust-threshold tuning
- new optimizer/LR
- alternate camera subset
- post-hoc Jacobian rebuild
- local GPU
- full training

## 9. Required outputs

VJS0_ENVIRONMENT.json
VJS0_SOURCE_IDENTITY.json
VJS0_CAMERA_SEMANTICS.json
VJS0_LINEARIZATION_PARITY.json
VJS0_RAW.json
VJS0_SUMMARY.json
VJS0_VERDICT.json
VJS0_REPORT.md
MANIFEST.sha256

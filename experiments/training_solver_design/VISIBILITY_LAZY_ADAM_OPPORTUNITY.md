# Full-resolution, visibility-lazy Adam: an opportunity test, not a winning method

2026-09-27. Goal remains shorter **complete** training wall time and higher
quality on genuinely untouched views than current strong 3DGS systems. This
document tests one systems mechanism without reducing input resolution,
dropping the original loss, or declaring a quality gain. It is not an
originality or SOTA claim.

## Measured opportunity and its ceiling

The pinned FastGS-Big room model uses 1600x1066 effective training images.
An isolated 4,099 -> 4,150 author-code continuation on A100 GPU 2 used
PyTorch profiler on 44 ordinary steps after the 4,100 densification event.
The profiler reports 326.148 ms self CUDA time in those steps. Its operator
totals assign 140.660 ms (43.13%) to raster backward, 72.627 ms (22.27%) to
Adam step, and 58.069 ms (17.80%) to raster forward. Kernel rows are nested
within operator rows and must **not** be added to them. The profiler perturbs
timing; these shares are an opportunity diagnostic, not a campaign speedup.
Even erasing the entire measured Adam cost would cap this ordinary-step
speedup at about `1/(1-0.2227)=1.286x`, before scoring, topology, load/save,
and the late FastGS stages that update only every 32/64 iterations.

Read-only full-resolution renders of 40 sampled **training** cameras gave:

| Saved room model | Points | Mean radii>0 fraction/view | Range/view | Never projected in the 40 views |
|---|---:|---:|---:|---:|
| 10K | 690,064 | 24.857% | 6.387%–39.692% | 10.937% |
| 30K | 566,856 | 27.664% | 7.350%–44.610% | 0.800% |

`radii>0` is the renderer's projected-radius/frustum proxy, not unoccluded
visibility and not proof that a row's Adam gradient is zero. The 10K/30K
saved states are from the completed paired baseline trajectory. Raw logs:
`fastgs_evidence/room_big_step_profile.log`,
`room_big_10k_visibility.log`, and `room_big_30k_visibility.log`.
In the same 40-camera sample order at 10K, gaps between repeated positive
projections of a Gaussian were 2/6/14 camera steps at the 50th/90th/99th
percentiles. This order is a diagnostic random sample, not the actual author
training camera order; see `room_big_10k_visibility_gaps.log`.

A separate 12-camera room@10K projection pilot used a deliberately loose
world-space sphere of `2 * 3 * max(scale) + 0.03` around each saved center,
with two pixels of angular frustum padding. It marked 34.295% of Gaussians
as possibly visible per camera versus 27.643% with positive raster radius,
and missed zero positive-radius cases in these 12 cameras. The fixed 0.03
world-unit allowance is a probe value, **not** a generally valid Adam drift
bound; the exact per-row moment bound below must replace it. This empirical
zero-false-negative sample is not a proof across states, cameras, or renderer
edge cases. Raw output: `fastgs_evidence/room_big_10k_sphere_probe.log`.

## Candidate: settle zero-gradient optimization debt on demand

For a Gaussian parameter row outside a camera's **conservatively expanded**
support, its current render contribution and photometric gradient are zero
under the pinned renderer's finite-support semantics. Dense Adam still
updates this row through its decaying momentum. Instead, maintain a logical
optimizer-step timestamp `s` and physically materialize `(theta,m,v)` only
before a camera might need the row, before any topology/opacity/score event
that reads its current parameters, and at export. Between materializations,
the exact dense-Adam zero-gradient transition is a deferred debt:

```text
m_j = beta1*m_(j-1)
v_j = beta2*v_(j-1)
theta_j = theta_(j-1) - eta_j*(m_j/(1-beta1^j))
          /(sqrt(v_j/(1-beta2^j)) + eps)
```

For regular FastGS optimizer steps 1..15K, `j` is the optimizer-step count.
After 15K, the author's gradients accumulate across 32/64 rendered views;
the debt clock advances only on actual Adam steps. A row with a nonzero
accumulated gradient must be settled and updated at that step. No skipped
render, pixel, or training gradient is allowed. Clones, splits, pruning,
opacity reset, SH-degree changes and checkpoints require exact transaction
ordering and a flush of affected rows. In particular, the author's score
reads current topology state at 100-step events, so it cannot see stale rows.

CPU FP64 `lazy_adam_oracle.py` compared 500 variable-learning-rate Adam
steps across 64 coordinates with approximately 25% nonzero coordinate
gradients against PyTorch Adam. After every deferred zero-gradient interval
was replayed, maximum final absolute errors were 4.44e-16 in parameters,
1.39e-17 in first moments and 0 in second moments. An initial failed check
was caused by accidentally converting the lazy reference to FP32 for the
comparison; that check failed at 1.18e-7 parameter error, then was corrected
without relaxing the 1e-12 gate. This proves only scalar recurrence parity
on CPU, not a production sparse optimizer.

## Main unresolved correctness and cost gates

**2026-09-27 follow-up and stage verdict.** A new fixed-checkpoint diagnostic,
`fastgs_screen_envelope_probe.py`, bounds the rendered screen-space ellipse
from the maximal post-drift scale, the source's 1.3-FoV Jacobian clamp,
the +0.3 covariance diagonal, the `mult=0.5` opacity threshold, and a
16-pixel tile plus 2-pixel numerical margin. It is a source-derived proxy,
not a formal floating-point proof. On room@4,099 after eight hypothetical
zero-gradient Adam steps, **all 272 training cameras** (651,547 Gaussians;
1600x1066) had zero false negatives against positive raster radius. Mean
`radii>0` fraction was 25.5201%; the new possible fraction was 31.6598%.
The prior `2*3-sigma` sphere missed three cases on 12 cameras. Full output:
`fastgs_evidence/room_big_4099_screen_envelope_all_train.log`. A first
40-camera run also had zero misses and 31.8867% possible; its log is retained.
An attempted `--views 311` run stopped before rendering because 311 is the
total image count, while only 272 are training cameras; the script now accepts
`--views 0` for all training cameras. No training behavior changed.

The author's `optimizer_step` makes 10,901 regular Adam calls from 4,100 to
15,000, 157 calls from 15,001 to 20,000, and 234 calls from 20,001 to
34,999: **11,292** calls in the compared 4,099->35K continuation. The
44-step early profiler assigns 72.627 ms to Adam, or 1.651 ms/call. If that
cost were representative of every call, *erasing all Adam work* saves only
18.64 s; even the unrealistically generous assumption that the 68.3402%
screen-envelope-negative rows cost zero and require no catch-up saves about
12.74 s. The observed 35K-vs-30K complete-wall gap is 29.06 s. An exact
deferred scheme must eventually repay zero-gradient Adam arithmetic and pay
for the visibility gate, metadata and topology flushes, so these figures are
optimistic opportunity estimates, not measured speedups or strict global
bounds. Late-stage 32/64-view gradient accumulation further narrows
per-render opportunity. This candidate has **not** earned an isolated GPU
implementation as the way to buy extra full-resolution steps and satisfy
both user goals. It remains a systems idea only if independent evidence
later establishes a much larger full-trajectory cost opportunity.

**2026-09-27 failure of the first geometric gate.** On the real room@4,099
checkpoint, eight hypothetical dense zero-gradient Adam steps changed
`radii>0` from false to true in 401 Gaussian-camera cases across 12 sampled
1600x1066 training views. The attempted stale-center sphere with an exact
per-row xyz drift bound and `factor * 3 * max(post-bound scale)` missed
256/16/3/0 positive-radius cases for factors 1/1.5/2/4; factor 4 admitted
39.206% of rows on average. The three factor-2 misses include one Gaussian
whose radius was already positive *before* the hypothetical updates. Thus the
failure is not solely Adam motion: the world-space 3-sigma sphere plus the
chosen screen padding does not conservatively model this renderer's
screen-space conic and tile-touching predicate. The author CUDA source uses
only a near-plane check in `in_frustum`, adds 0.3 to each screen-space
covariance diagonal,
and determines positive radius after `duplicateToTilesTouched`; none of these
are captured exactly by the sphere proxy. Even factor 4's zero misses in this
sample are not a correctness proof. Raw evidence is
`fastgs_evidence/room_big_4099_momentum_visibility_debug.log` and the
executed script `fastgs_momentum_visibility_probe.py`. Do not implement lazy
GPU updates or claim unchanged quality until a renderer-semantic conservative
gate survives trajectory-scale tests.

An unmaterialized Gaussian may move or change scale under momentum and
become visible in a new camera. Testing frustum culling on stale geometry
would silently change the rendered training image. A correct implementation
must first bound *every* zero-gradient update since `s`. If learning rates
over the deferred interval are at most `eta_max`, current moment state is
`m_s,v_s`, and there are `k` skipped optimizer steps, one conservative
per-coordinate displacement bound is:

```text
abs(delta_theta) <= eta_max*abs(m_s)/(1-beta1^(s+1))
  * [beta1*(1-beta1^k)/(1-beta1)]
  / [sqrt(beta2^k*v_s/(1-beta2^(s+k))) + eps]
```

This uses the first skipped step's largest bias-correction factor and the
last skipped step's smallest RMS denominator. The scalar inequality passed
10,000 randomized FP64 cases with varying step count, gap, moment scale and
learning-rate sequence; the largest actual-displacement/bound ratio was
1.0000000000000002 within floating-point tolerance. See
`lazy_adam_oracle.json`. Apply bounds to xyz and log
scales, then use the largest possible ellipsoid radius to expand the
camera-frustum test; rotation cannot enlarge that enclosing sphere. The
actual FastGS finite support, projection conventions, camera matrices and
alpha cutoff must be checked in code, not inferred from this sketch. If the
expanded support is too loose and selects almost all Gaussians, the method
has no opportunity. A periodic full flush can bound `k` and metadata cost
but must itself be included in training wall time.

Next gates, in order:

1. Independently test the displacement bound against dense PyTorch Adam
   with varying learning rates and real optimizer moments, including
   opacity resets and newborn states. Check that conservative projection
   has **zero false negatives** on full-resolution training cameras.
2. On a frozen real FastGS trajectory, report possible-visibility fraction,
   catch-up gap distribution, extra per-row arithmetic, state memory,
   complete flush cost, and the maximum reclaimable Adam time. If that
   cannot produce a material end-to-end gain, stop before CUDA work.
3. Only if gates 1–2 pass, implement an isolated GPU optimizer and require
   parameter/moment/render/gradient parity against the author trainer
   through densification, pruning, and 30K. Then compare full wall time.
4. Exact optimizer semantics alone produce the **same 30K quality**. A
   separate validated quality mechanism or extra full-resolution training
   within a still shorter wall budget is required to meet the user's joint
   target. The 35K development continuation below probes whether extra
   steps could help on room; it does not prove the method.

## Unmodified extra-step quality probe

The pinned author trainer continued the common room@4,099 checkpoint to
35K on A100 GPU 4 and saved both 30K and 35K states. Both used the same
1600x1066 input, author loss, renderer, scoring and optimizer. Its built-in
training timer reported 229.2903 s for 4,099 -> 35K and the complete
process wall timer 291.46 s. The author metrics script evaluated the same
39 *development* views from this single trajectory:

| Snapshot | PSNR dB | SSIM | LPIPS |
|---|---:|---:|---:|
| 30K | 32.229099 | .9303260 | .1887624 |
| 35K | 32.265259 | .9305426 | .1883504 |

The extra 5K steps moved aggregate metrics by +0.036160 dB PSNR,
+0.000217 SSIM and -0.000412 LPIPS; 31/39, 23/39 and 28/39 respective
views improved. The 30K checkpoint here is within 0.000851 dB PSNR of
the separate 30K baseline continuation, but runs are not bitwise identical.
The separate 4,099 -> 30K baseline took 192.0766 s built-in and 262.40 s
full process wall. Relative to that run, the 35K process cost 29.06 s
more wall time; an exact-lazy 35K method would need to save over 9.97% of
its **complete** wall time merely to finish before that 30K reference. The
single-scene metric increase is small and unconfirmed across seeds, and room
views were used for development. No lazy GPU implementation or speed result
exists. Evidence: `fastgs_evidence/room_big_branch_base35k.log`,
`room_big_branch_base35k_results.json`, and `room_big_branch_base35k_per_view.json`.

## Prior-art boundary

[Faster-GS](https://arxiv.org/html/2602.09999v1) Sec. 3.3 already develops
custom fused Adam and fuses the optimizer into raster backward. It explicitly
notes that simply skipping invisible-row updates can cause inconsistencies
and regressions. Its Table 3 reports, on five outdoor Mip-NeRF 360 scenes
with RTX 4090, 5m04s/24.73 dB for fused updates and 3m03s/24.59 dB when
invisible updates are skipped: a large *author-reported* time opportunity
with a 0.14 dB quality loss. Exact deferred updates would seek that saved
memory traffic without accepting the changed Adam trajectory, but neither
similar speed nor recovered quality has been demonstrated here. It should
ultimately be tested on the Faster-GS system as well as FastGS-Big, since
the two trainers have different optimizer and visibility behavior.
The official [Faster-GS source](https://github.com/nerficg-project/faster-gaussian-splatting)
is already locally mirrored at `public_baselines/faster-gs` at
`117bdbacfd7df6c1f06d9d959b1f5ab02d4d3847`; its `main` branch contains
the custom `FusedAdam`, and its README identifies a separate
`FasterGSFused` branch for backward/optimizer fusion. That branch has not
yet been source-audited in this workspace.
[Taming 3DGS](https://arxiv.org/abs/2406.15643) uses sparse
updates, and [BOGausS](https://doi.org/10.23919/EUSIPCO63237.2025.11226626)
uses per-Gaussian optimizer age. Neither the Adam recurrence nor generic
fusing/sparse updates are original here. A possible new contribution is
**conservative visibility plus exact deferred dense-Adam settlement** in a
topology-changing 3DGS trainer, if a complete source audit confirms priority
and the GPU/full-training gates pass. Until then this remains a hypothesis.

[ACE-GS](https://arxiv.org/html/2606.21244) Sec. 4 separately claims
gradient–momentum cosine densification, opacity-gradient sensitivity pruning,
and FFT high-frequency residual injection. Those mechanisms are not part of
this candidate and cannot be recast as novel extensions. Its author Table 1
reports 5m30s/28.10 dB/.821/.215 on Mip-NeRF 360 on RTX 4090, but lacks
FastGS-Big in that table; its seconds cannot be compared to our A100 room
numbers. The HTML paper contains no project-code hyperlink identified in
this audit, which does not prove code is unavailable elsewhere.

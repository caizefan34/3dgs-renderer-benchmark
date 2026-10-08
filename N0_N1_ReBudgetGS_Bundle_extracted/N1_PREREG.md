# N1 — ReBudgetGS Mechanism Measurement Preregistration

Frozen date: 2026-09-28

## Purpose

N1 is a **mechanism measurement**, not the final method.

It asks whether post-densification 3DGS training contains enough variation in update value that reallocating a fixed backward budget between:

- skip (`b=0`);
- single-view (`b=1`);
- rare paired-view (`b=2`);

can outperform any fixed action policy in **quality progress per unit wall time**.

Unlike earlier mechanism-first campaigns, N1 does not require a spectacular oracle win to continue.
It classifies the opportunity size and identifies the cheapest usable signal.

---

# 1. Frozen baseline

Use frozen P12 C0 identity and training semantics.

Primary scenes:
- room
- bicycle

Post-densification phase only:
- 15000 → 30000

No garden in N1.

Use real P12/C0 checkpoints:
- 15K
- 20K
- 25K

Do not use 30K as a primary state because it is already the endpoint target.

At each checkpoint restore:
- Gaussian parameters;
- Adam moments;
- optimizer/global-step state;
- SH degree;
- camera sequence metadata.

---

# 2. Probe states

Per scene and checkpoint:

- sample 64 deterministic anchor opportunities;
- total 192 anchor states per scene;
- 384 total.

For each anchor state:

1. select anchor training view `v1` from a deterministic frozen list;
2. select a pool of 7 candidate partner views `v2..v8`;
3. candidate pool must include:
   - 3 high camera-overlap candidates;
   - 2 medium-overlap candidates;
   - 2 low-overlap candidates,
   based only on camera/visibility geometry available before the probe;
4. do not select partners using their current gradients.

The model state is reset to the identical checkpoint before every counterfactual action.

---

# 3. Ground-truth action evaluation

Evaluate actions:

## A0 — skip
- perform the same forward/loss needed by a real gating policy;
- no backward;
- no optimizer update.

## A1 — single-view
- ordinary P12/C0 backward on `v1`;
- ordinary Adam update.

## A2-random
- `v1` + one deterministic random partner from the pool;
- aggregate their true gradients as an equal-weight two-view minibatch;
- one optimizer update.

## A2-best
Diagnostic oracle only:
- evaluate each allowed partner;
- retain the pair giving the best held-out progress-per-time.
- NEVER usable by the eventual method.

A2-best measures the maximum value of having an additional view.
It is not the production policy.

---

# 4. Held-out progress target

For every action evaluate immediately after the counterfactual update on a fixed held-out set of 8 training cameras that excludes all action views.

Record:

- mean held-out L1;
- mean held-out fused training loss;
- optional PSNR;
- parameter update norm.

Define:

\[
\Delta L(a) = L_{heldout,before} - L_{heldout,after(a)}
\]

and efficiency:

\[
E(a)=\frac{\Delta L(a)}{T(a)}
\]

where `T(a)` is clean measured action wall time including all required forward/backward/update work.

Because one-step noise can produce negative `ΔL`, report medians, trimmed means, and sign rates.

---

# 5. Real timing

Timing must be measured from the same restored state using warm-up and repeated action execution.

For each action type:
- 10 warm-ups;
- 30 timing repeats on frozen state;
- CUDA events + host wall time;
- no profiler during primary timing.

Record separately:
- forward;
- loss;
- backward;
- optimizer;
- total.

The final method must count gating overhead.

---

# 6. Historical signals

Construct only from information a real online method could have.

For `v1`:
- current forward loss;
- loss / EMA-loss ratio;
- loss slope;
- iterations since last backward;
- number of times backwarded in last 512 opportunities.

For every view maintain from its most recent true backward:
- a compact gradient sketch `s_v`;
- sketch age.

Primary sketch:
- fixed seeded Rademacher projection;
- dimension 256;
- computed separately for:
  - xyz;
  - opacity;
  - scale+rotation;
  - SH;
- concatenate normalized group sketches.

The projection seed is frozen globally.
No learned projection in N1.

Optional geometry signals:
- camera-center distance;
- view-direction angle;
- visibility/Jaccard overlap if available cheaply from cached visibility metadata.

---

# 7. Ground-truth conflict labels

For diagnostic purposes only, compute current true gradients for the anchor and each candidate partner.

Report:
- cosine(g1, g2) by parameter group;
- global normalized cosine;
- gradient norm ratio;
- support/visibility overlap.

This current partner gradient is **not** allowed as a production feature.

Measure whether stale compact sketches predict current conflict.

Primary correlation target:
- Spearman correlation between predicted stale-sketch disagreement and current true gradient disagreement.

---

# 8. Opportunity metrics

N1 tests three separate claims.

## H1 — skip opportunity exists
At least 20% of anchor states should have:
- A0 within 10% of A1 held-out progress impact after normalizing for one-step noise,
  OR A1 has non-positive held-out improvement.

This is descriptive; SkipGS already establishes the general phenomenon.

## H2 — paired-view rescue is sometimes materially better
At least 15% of anchors in each scene must satisfy BOTH:
- `E(A2-best) >= 1.20 * E(A1)` when A1 improvement is positive;
- absolute held-out improvement of A2-best exceeds A1 by a preregistered noise floor estimated from repeated A1 replicas.

This is the key new mechanism signal.

## H3 — action heterogeneity
Across each scene:
- at least 15% anchors prefer A0 under efficiency;
- at least 30% prefer A1;
- at least 10% prefer A2-best.

If a single action dominates >90% of anchors, dynamic allocation is weakly motivated.

---

# 9. Budget-reallocation simulation

Using measured action outcomes/costs only, solve an offline constrained assignment:

\[
\max \sum_i \Delta L_i(a_i)
\]

subject to:

\[
\sum_i C_i(a_i) \le \rho \sum_i C_i(A1),
\]

for:
- rho = 0.60
- rho = 0.75
- rho = 0.90

This is an oracle upper bound and must be labelled `OFFLINE_ASSIGNMENT_UPPER_BOUND`.

Compare with:
- all A1;
- best global fixed mixture of A0/A1;
- best global fixed mixture of A0/A1/A2-random.

The important quantity is not raw oracle speedup, but whether **reallocation** has additional value beyond 0/1 skipping.

Strong reallocation opportunity:
- at rho=0.75, oracle assigned held-out progress >= 95% of all-A1 progress;
- and >=10% better progress than the best 0/1 mixture at the same measured cost;
- in both scenes.

---

# 10. Cheap-policy diagnostic

Only if strong reallocation opportunity exists.

Freeze a non-neural policy family:

Inputs:
- loss/EMA ratio;
- backward age;
- stale-sketch disagreement;
- sketch age;
- geometric overlap.

Model:
- multinomial logistic regression, L2, C=1.0;
- no hyperparameter search.

Cross-scene:
- train room -> test bicycle;
- train bicycle -> test room.

Evaluate under the same rho=0.75 budget.

Policy is promising if in both directions:
- held-out progress >=90% of the offline assignment upper bound;
- and >=7.5% above best 0/1 mixture at matched cost.

This does not authorize a final paper claim.
It only authorizes N2 implementation.

---

# 11. Verdicts

## `N1_STRONG_REALLOCATION_SIGNAL`
Both scenes:
- H2 passes;
- H3 passes;
- strong rho=0.75 reallocation opportunity passes;
- cheap-policy diagnostic passes both cross-scene directions.

Authorize:
`N2_REBUDGETGS_TRAINING_PROTOTYPE`

## `N1_MECHANISM_EXISTS_POLICY_WEAK`
Both scenes have H2/H3 + reallocation opportunity, but cheap transfer policy fails.

Interpretation:
the mechanism is real but the current cheap predictor is inadequate.
One additional signal-design iteration is allowed, but no threshold fishing.

## `N1_SMALL_BUT_ACTIONABLE`
At least one scene shows >=5% matched-cost improvement over 0/1 skipping and no catastrophic counterevidence.

Interpretation:
may still support a smaller publishable optimization; do not kill solely for missing 3x-scale headroom.

## `N1_NO_REALLOCATION_VALUE`
Paired-view rescue adds <5% matched-cost value over best 0/1 allocation in both scenes.

Close ReBudgetGS as a distinct mechanism.
Do not disguise SkipGS reproduction as a new method.

---

# 12. Required artifacts

- `N1_SOURCE_IDENTITY.json`
- `N1_CHECKPOINT_IDENTITY.json`
- `N1_PROBE_LIST.json`
- `N1_ACTION_RESULTS.parquet`
- `N1_ACTION_TIMING.json`
- `N1_GRADIENT_CONFLICT.parquet`
- `N1_SKETCH_PREDICTIVITY.json`
- `N1_ACTION_HETEROGENEITY.json`
- `N1_BUDGET_FRONTIER.json`
- `N1_CROSS_SCENE_POLICY.json` if gated
- `N1_CONTAMINATION.json`
- `N1_VERDICT.json`
- `N1_REPORT.md`
- `MANIFEST_N1.json`

No final method implementation.
No garden.
No full 15K→30K training campaign in N1.

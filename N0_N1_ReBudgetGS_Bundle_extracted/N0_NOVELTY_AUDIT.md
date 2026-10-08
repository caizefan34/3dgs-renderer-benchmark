# N0 — ReBudgetGS Novelty & Mechanism Audit

Frozen date: 2026-09-28

## Candidate working name

**ReBudgetGS — Conflict-Aware Backward Budget Reallocation for 3D Gaussian Splatting**

This is a working name, not a novelty claim.

## Core problem

Standard 3DGS spends approximately one backward pass per sampled training view.

Existing work shows two apparently opposing facts:

1. Some post-densification views provide diminishing update value and can be skipped.
2. Multi-view gradients can reduce stochastic-gradient variance and cross-view inconsistency.

ReBudgetGS asks:

> Given a limited backward-compute budget, should the next training opportunity receive 0, 1, or 2 real-view backward passes?

The intended action set is:

- `b=0`: forward/statistics only, no backward/update;
- `b=1`: ordinary single-view backward/update;
- `b=2`: pair two real training views and form one joint update.

Crucially, the average backward-view count is constrained:

\[
E[b_t] < 1
\]

during the post-densification phase.

Thus `b=2` is not extra compute added on top of the baseline. It is paid for by compute saved on `b=0` opportunities.

## Candidate mechanism

Maintain a compact historical optimization state per view:

- recent loss / EMA loss;
- last-backward age;
- compact gradient sketch from the most recent real backward;
- optional visible-Gaussian overlap summary.

Use the current view plus historical state to estimate:

1. **diminishing-return score** — whether this view can safely receive `b=0`;
2. **cross-view conflict / uncertainty score** — whether this opportunity deserves `b=2`;
3. otherwise use `b=1`.

The method must not modify gradient direction as HarmoGS does.
The method must not use future ground truth or an oracle during real training.

---

# Prior-art boundary

## Efficient Multi-view Training for 3DGS (2025)

Relevant overlap:
- identifies high variance of single-view stochastic gradients;
- shows multi-view mini-batches reduce variance;
- proposes efficient partial rendering and multi-view density control.

Boundary:
- its central mechanism is efficient multi-view training, not a constrained online controller that dynamically allocates `0/1/2` backward-view cardinality per update opportunity;
- ReBudgetGS must not claim "first multi-view 3DGS training" or "first to reduce gradient variance using multiple views."

## SkipGS (ECCV 2026)

Relevant overlap:
- post-densification phase;
- per-view training state;
- view-adaptive backward gating;
- skips low-value backward passes.

Boundary:
- SkipGS is fundamentally `0/1` backward gating;
- ReBudgetGS must demonstrate value beyond a strong reimplementation of SkipGS;
- the novelty window is **reallocation**, not skipping:
  compute saved from low-value opportunities is deliberately spent on rare multi-view rescue updates under an average backward budget below baseline.

If `0/1` alone matches ReBudgetGS, the new method is not justified.

## HarmoGS (2026)

Relevant overlap:
- computes two view-specific gradients;
- explicitly measures/handles cross-view gradient conflict.

Boundary:
- HarmoGS targets in-the-wild robustness and modifies conflicting gradients through harmonization;
- ReBudgetGS targets compute allocation under a budget;
- it must not claim first use of gradient conflict in 3DGS.

If the method requires changing/rotating gradient direction to work, it has drifted into HarmoGS territory.

## S2Gaussian (CVPR 2025)

Relevant overlap:
- stores historical per-Gaussian gradient trend;
- uses gradient-direction consistency for robust optimization.

Boundary:
- ReBudgetGS cannot claim first persistent gradient-history state;
- novelty must be the use of compact historical optimization state to allocate backward-view cardinality under a compute budget.

## GDAGS (ICLR 2026)

Relevant overlap:
- gradient direction/coherence;
- uses coherence to alter density control.

Boundary:
- ReBudgetGS does not use gradient coherence to control densification.

## Next-Best-View / Fisher-guided view selection

Relevant overlap:
- scores informative views;
- uses information gain/Fisher signals.

Boundary:
- these methods select/acquire/prioritize views;
- ReBudgetGS uses the existing fixed training set and decides **how much backward compute** an update opportunity receives.

It must not claim first information-aware view selection.

## 3DGS-LM / Opt3DGS

Relevant overlap:
- improve late-stage optimization/convergence.

Boundary:
- ReBudgetGS is an adaptive compute-allocation policy, not a new second-order optimizer.

---

# N0 novelty verdict

**N0_NOVELTY_WINDOW_PLAUSIBLE_BUT_NARROW**

The broad ideas below are NOT novel:
- multi-view training;
- gradient conflict;
- per-view loss history;
- backward skipping;
- view selection;
- persistent gradient trends.

The potentially publishable mechanism is the conjunction:

> **budget-constrained online reallocation of backward compute across training opportunities, using compact historical optimization state to choose among `0/1/2` real-view updates, with rare multi-view rescue paid for by skipped backward work.**

This remains a novelty hypothesis until a deeper citation/code audit is done before submission.

---

# Publication requirement

A paper cannot be based merely on a controller heuristic.

At minimum it must show:

1. a measurable heterogeneous optimum among `b=0/1/2`;
2. reallocation beats both:
   - dense single-view baseline,
   - strong `0/1` SkipGS-style baseline;
3. strict-quality parity across multiple scenes;
4. actual E2E wall-time gain;
5. a principled compact signal with cross-scene transfer;
6. ablation showing `b=2` rescue is necessary rather than decorative;
7. compatibility with a strong renderer/training baseline.


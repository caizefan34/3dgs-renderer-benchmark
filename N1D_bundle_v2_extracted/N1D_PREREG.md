# N1D — ReBudgetGS Mechanism Disambiguation

Frozen date: 2026-09-29

## Goal

Determine whether the small A2 value observed in N1 is caused by **genuine second-view information** or can be explained by **single-view update damping**.

N1D is a targeted diagnostic. It is not a final method and does not change the old N1 verdict.

## Scope

- Frozen P12 C0.
- room + bicycle.
- checkpoints 15K / 20K / 25K.
- reuse the exact 64 anchors/checkpoint/scene from N1.
- reuse the same held-out sets.
- no garden.
- no full training.

## Partner freeze

For every anchor, before new action execution, freeze three deterministic random partners from the original 7-view pool using a new globally fixed seed.

Do not select partners using action outcomes, current gradients, or overlap tier.

## Actions

All actions start from an identical exact restored state.

### A0
No backward/update.

### A1-1.00
Original single-view C0 action.

### A1-0.75
Identical A1 gradient and Adam semantics, but multiply the parameter-group learning rates for this one step by 0.75, then restore the frozen schedule.

### A1-0.50
Same with one-step LR multiplier 0.50.

### A1-0.25
Same with one-step LR multiplier 0.25.

These are diagnostic damping controls, not proposed methods.

### A2-r1 / A2-r2 / A2-r3
Two true full-resolution view gradients, equal-weight mean, one ordinary Adam update, using three independently frozen random partners.

No A2-best oracle is needed in N1D.

## Measurements

Use the same 8-camera held-out metric as N1:

`DeltaL = L_before - L_after`.

Primary action cost must be measured with the same timing hygiene as N1.

For A2 define per anchor:

- `A2_median`: median DeltaL across r1/r2/r3;
- `A2_mean`: mean DeltaL across r1/r2/r3.

For the damping family define:

- `D_best`: best of {A0, A1-1.00, A1-0.75, A1-0.50, A1-0.25} for diagnostic upper-bound analysis;
- each fixed alpha is also reported separately.

## Core tests

### D1 — random-partner reproducibility

A2 is not considered a real mechanism if the original single-random benefit disappears when evaluated as the median of three independently frozen random partners.

### D2 — damping equivalence

For each scene compute a matched-cost offline frontier with:

- damping-only action set: A0 + A1 alpha-grid;
- A2 action set: A0 + A1-1.00 + A2_median.

If A2 adds <3% progress over the damping-only frontier at rho=.75 in both scenes, conclude:

`N1D_A2_EXPLAINED_BY_DAMPING_OR_NOISE`.

### D3 — unique second-view value

To support ReBudgetGS, both scenes must show at rho=.75:

- A2 frontier >=5% above damping-only frontier;
- at least 10% of anchors where A2_median exceeds D_best by more than the repeated-A1 numerical noise floor;
- effect not confined to a single checkpoint.

Then conclude:

`N1D_SECOND_VIEW_VALUE_CONFIRMED`.

### Borderline

If one scene passes and one misses, or gain is 3–5%:

`N1D_WEAK_SECOND_VIEW_VALUE`.

This may justify one sequential validation but not N2 implementation.

## Sequential-validity caution

Even `N1D_SECOND_VIEW_VALUE_CONFIRMED` does not prove a full training controller works.
The next required phase would be a short sequential rollout comparing:

- dense A1;
- official SkipGS-style 0/1;
- fixed/global 0/1/2 mixture;
- a frozen cheap ReBudgetGS policy.

No full 30K campaign is authorized by N1D.

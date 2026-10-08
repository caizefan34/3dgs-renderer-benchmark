# P17-P0 Provenance Repair

## Status: REPAIRED

## Problem

The remote `P17_UNITS/` directory was overwritten by the tau=4 sensitivity run
(last in `run_all.sh` order: main(τ=2) → τ1 → τ4). The final state of the
directory holds tau=4.0 unit files, not the primary tau=2.0 units.

The primary aggregation (`P17_GRADIENT_AGG.json`), step aggregation
(`P17_STEP_AGG.json`), work oracle (`P17_WORK_ORACLE.json`), verdict
(`P17_VERDICT.json`), and main logs (`room_main.log`, `bicycle_main.log`)
were all preserved intact.

## Repair Method

Primary tau=2.0 unit data was reconstructed from the frozen main logs using
the same regex parsing logic as `make_aggs.py`:

```
<scene> cam<N>: C_oracle=<C> n_d2+=<n> means_cos=<mc> R_step=<R> P_audit=<P>
```

Each reconstructed unit file carries:
- `reconstruction_tag: "RECONSTRUCTED_FROM_FROZEN_PRIMARY_LOGS"`
- `reconstruction_source: "<scene>_main.log"`

These are **not** original unit artifacts. The original tau=2.0 unit files
wrote more detail (full per-param metrics, n_visible counts, G_dense/G_hyb,
cost_by_d, L_before/L_after, P_audit_per_cam) which is not recoverable from
the log summary lines. The reconstructed files contain only the fields that
appear in the log line plus the known constants (tau=2.0, d1_parity_pass=true,
reference_improving=true).

## Verification

| Check | Result |
|-------|--------|
| Room C_oracle median (recon) | 0.09065 ✓ matches frozen |
| Room C_oracle values (12/12) | all match `P17_WORK_ORACLE.json` |
| Bicycle C_oracle median (recon) | 0.03215 ✓ matches frozen |
| Bicycle C_oracle values (12/12) | all match `P17_WORK_ORACLE.json` |
| Room R_step + P_audit (12/12) | all match `P17_STEP_AGG.json` (1e-10 tol) |
| Bicycle R_step + P_audit (12/12) | all match `P17_STEP_AGG.json` (1e-10 tol) |

## Verdict

**UNCHANGED: `P17_P0_HARD_DROP`**

Hard drop triggers (both scenes):
- `means_cos < 0.98` (median ~0.0005 room, ~0.0004 bicycle)
- `R_step < 0.8` (median ~−0.015 room, ~−0.002 bicycle)

The reconstruction does not alter any gate values.

## File Inventory

### Reconstructed (this repair)
```
P17_UNITS_PRIMARY_RECONSTRUCTED/room/camera_{12,38,64,90,116,142,168,194,220,246,272,298}.json
P17_UNITS_PRIMARY_RECONSTRUCTED/bicycle/camera_{8,24,40,56,72,88,105,121,137,153,169,185}.json
```
All 24 files tagged `RECONSTRUCTED_FROM_FROZEN_PRIMARY_LOGS`.

### Frozen originals (untouched)
```
P17_GRADIENT_AGG.json       (primary τ=2 aggregation)
P17_STEP_AGG.json           (per-camera R_step + P_audit)
P17_WORK_ORACLE.json        (per-camera C_oracle)
P17_VERDICT.json            (P17_P0_HARD_DROP)
P17_SENSITIVITY_TAU.json    (τ=1,4 sensitivity)
P17_CAMERA_SET.json
P17_SOURCE_IDENTITY.json
P17_STATE_IDENTITY.json
P17_D1_PARITY.json
P17_CONTAMINATION.json
room_main.log
bicycle_main.log
room_tau1.log
bicycle_tau1.log
room_tau4.log
bicycle_tau4.log
P17_UNITS/                  (final state: τ=4 files — overwritten)
```

## Provenance Map

See `P17_PROVENANCE_MAP.json` for the complete file-to-source mapping.

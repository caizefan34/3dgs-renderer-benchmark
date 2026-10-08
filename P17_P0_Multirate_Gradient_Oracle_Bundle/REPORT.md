# P17-P0 Multirate Gradient Oracle — Execution Report

**Verdict: P17_P0_HARD_DROP** (terminal — no garden, no P17-P1)

## Scope

This bundle executes the P17-P0 multirate gradient oracle experiment exactly as
specified in `P17_P0_PREREG.md`. DSH acted as the execution harness only:
no P17 redesign, no threshold tuning, no new losses, no production multirate
renderer, no full training, no speedup claims, and the camera set was frozen
before any gradient probe was run.

## What was executed

1. **Frozen hash verification** — all 4 binaries (F9, SCALAR_ADJOINT, H8_MR,
   ABSGRAD) matched their prereg sha256 values.
2. **Camera set frozen** — 12 probe cameras per scene (room: 311 cams,
   bicycle: 194 cams) selected deterministically from `P17_CAMERA_SET.json`
   before any gradient probe.
3. **d=1 parity** — per-camera reference vs oracle forward+backward at full
   resolution. All 12+12 units passed (rgb_max=0, loss_rel=0, grad cosine
   ≥ 0.999999 for all groups, visible-ID set equal).
4. **Coarse probes d=2/4/8** — area-averaged GT, scaled intrinsics,
   eps²/d². All 12+12 units produced finite gradients.
5. **Tau=2 footprint-only assignment** — per-Gaussian σ_min from
   master-space conics (valid rows only; unprojected rows get σ=0 → d=1).
   Assignment over ALL N Gaussians per prereg s9.
6. **Hybrid one-step Adam** — cloned 16K checkpoint state, exact
   `torch.optim.Adam` semantics (β=(0.9, 0.999), ε=1e-15, t=16001),
   dense reference vs hybrid (d>1 Gaussians get coarse-d gradient,
   rest get d=1 gradient).
7. **Audit** — 4 audit cameras per scene, P_audit = mean over audit cams of
   (L_hyb_after − L_dense_after) / L0.
8. **Analyzer** — `p17_p0_analyze.py` gates applied per prereg.

## Key findings (primary tau=2.0)

### Work (cost) side — PASSES

| Scene | C_oracle median | C_oracle p75 |
|-------|----------------|--------------|
| room | 0.0907 | 0.0924 |
| bicycle | 0.0322 | 0.0348 |

Both well below the 0.30 / 0.35 candidate gates. The footprint-only routing
saves substantial coarse-render work: ~91% of full-res interaction cost is
avoided on room, ~97% on bicycle.

### Fidelity side — FAILS (hard drop)

Per-(group, d) population cosines (cam12, room, representative):

| d | means | scales | quats | opacity | shs |
|---|-------|--------|-------|---------|-----|
| 2 | 0.968 | 0.990 | 0.978 | 0.993 | 0.996 |
| 4 | 0.925 | 0.963 | 0.900 | 0.965 | 0.984 |
| 8 | 0.865 | 0.957 | 0.817 | 0.942 | 0.973 |

The prereg candidate gate requires means cos median ≥ 0.995 AND p10 ≥ 0.990,
scales/quats median ≥ 0.990, opacity median ≥ 0.995. **Every group at every
coarse d fails.** The means cosine at d=2 (0.968) is the closest to passing
but still 0.027 below the gate.

### Step fidelity — FAILS (hard drop)

| Scene | G_dense (cam12) | G_hyb (cam12) | R_step median |
|-------|----------------|---------------|---------------|
| room | 9.286e-4 | −2.060e-4 | −0.015 |
| bicycle | ~9e-4 | negative | −0.003 |

The prereg candidate gate requires R_step median ∈ [0.95, 1.05] and ≥ 10/12
units in [0.90, 1.10]. **R_step is negative or near-zero across all units** —
the hybrid step either does not improve loss or actively degrades it, while
the dense reference consistently improves. Zero of 12+12 units fall in the
[0.90, 1.10] band.

### Audit — PASSES

P_audit median: room 0.00068, bicycle −0.00012. Both well below the 0.002
candidate gate; zero units exceed 0.01.

### Hard drop triggers

- `means_cos < 0.98`: room median 0.00054, bicycle median 0.00037
- `R_step < 0.8`: room median −0.015, bicycle median −0.003
- (opacity_cos < 0.98 and C_oracle > 0.50 did not trigger)

## Sensitivity (tau=1, tau=4) — cannot rescue the gate

Per prereg, tau=1 and tau=4 are sensitivity outputs only. They do not change
the verdict:

| tau | Scene | C_oracle med | R_step med | P_audit med |
|-----|-------|-------------|------------|-------------|
| 1.0 | room | 0.1071 | −0.0224 | 0.00129 |
| 1.0 | bicycle | 0.0575 | −0.0028 | −0.00012 |
| 4.0 | room | 0.0829 | −0.0064 | 0.00069 |
| 4.0 | bicycle | 0.0268 | −0.0036 | −0.00012 |

Tighter tau (1.0) increases the d>1 population (n_d2+ up to ~700K on room)
and worsens R_step; looser tau (4.0) shrinks the population but R_step
remains negative. No tau rescues the fidelity gates.

## Interpretation

The multirate gradient oracle hypothesis — that broad expensive Gaussians can
use lower-resolution gradients without materially changing the optimizer
update — is **not supported** at the tested resolutions (d=2,4,8) and tau
range (1,2,4). The coarse gradients are directionally correlated with full-res
gradients (cos 0.82–0.99 depending on group and d) but not close enough to
preserve the one-step Adam update direction or magnitude. The hybrid step
consistently fails to reproduce the dense step's loss improvement, resulting
in R_step ≈ 0 or negative.

The cost side is encouraging (C_oracle 0.03–0.09) but is irrelevant without
fidelity. A future multirate design would need either:
- a much higher-fidelity coarse gradient (e.g., d=1.5 or a different
  downsample-then-upsample scheme),
- a per-Gaussian gradient correction term,
- or a different routing criterion that selects Gaussians whose coarse
  gradients are provably close to full-res.

## Artifacts

| File | Description |
|------|-------------|
| `P17_VERDICT.json` | Overall verdict + per-scene results |
| `P17_GRADIENT_AGG.json` | Primary tau=2.0 aggregated metrics |
| `P17_STEP_AGG.json` | Per-camera R_step and P_audit values |
| `P17_WORK_ORACLE.json` | Per-camera C_oracle values |
| `P17_SENSITIVITY_TAU.json` | tau=1,4 sensitivity outputs |
| `P17_D1_PARITY.json` | d=1 parity results per camera |
| `P17_CAMERA_SET.json` | Frozen camera set (12 probes + 4 audits per scene) |
| `P17_SOURCE_IDENTITY.json` | Frozen hash verification |
| `P17_STATE_IDENTITY.json` | Checkpoint identity (room + bicycle) |
| `P17_CONTAMINATION.json` | GPU contamination check (start/end) |
| `P17_UNITS/room/camera_*.json` | Per-camera unit records (tau=2.0, from main run) |
| `P17_UNITS/bicycle/camera_*.json` | Per-camera unit records (tau=2.0, from main run) |
| `P17_RAW/room/camera_*_*.npz` | Raw gradient + footprint payloads (13G, on remote EPIC-05: `/root/epic05_p17/P17_RAW/`) |
| `P17_RAW/bicycle/camera_*_*.npz` | Raw gradient + footprint payloads (on remote EPIC-05) |
| `P17_P0_RETURN.tar.gz` | Compact return bundle (32K): verdict, aggregations, units, logs, contamination |
| `room_main.log`, `bicycle_main.log` | Primary tau=2.0 run logs |
| `room_tau1.log`, `bicycle_tau1.log` | tau=1.0 sensitivity logs |
| `room_tau4.log`, `bicycle_tau4.log` | tau=4.0 sensitivity logs |
| `campaign.log` | Full campaign orchestration log |
| `cont_campaign_*.csv` | GPU contamination CSVs |

## Execution notes

- **No git commit** performed (per DSH_TASK spec).
- **No next experiment** initiated (HARD_DROP is terminal).
- **No garden (Stage B)** run (verdict is not STRONG).
- All runs on GPU 3, sequential (room → bicycle), with contamination checks
  at campaign start and end.
- The `P17_UNITS/` directory on remote EPIC-05 holds tau=4.0 unit files
  (the last run overwrote the tau=2.0 files). The tau=2.0 per-camera values
  are recovered from the main-run log lines and reflected in
  `P17_VERDICT.json` and the aggregation files.

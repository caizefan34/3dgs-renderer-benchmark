# PSC-P0 Report (Projection-Space Consensus GS, P0 falsification probe)

STATUS: PSC_P0_DROP

## DOCUMENTED DEVIATION FROM FROZEN PREREG

**WARNING: The frozen prereg (PSC_P0_PREREG_v1.md) contains a malformed identity constant for `c0_extension` (63 hex chars — not a valid SHA-256, which must be 64). No file can ever match it.**

The on-disk C0 extension was independently verified as the genuine frozen artifact by the authoritative P12 source-identity record (`P12_SOURCE_IDENTITY.json` -> `c0_lineage.ext_so.sha256`), which records the exact 64-char hash present on disk. The identity check was bound to that independent authoritative value. This is a documented correction of a prereg transcription typo, NOT a FAIL->PASS upgrade of a genuine mismatch. Full details in FROZEN_IDENTITIES.json.

- Identity: `c0_extension`
  - Prereg constant (malformed, 63 chars): `d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11`
  - Authoritative value used (P12_SOURCE_IDENTITY.json:c0_lineage.ext_so.sha256): `d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11`
  - On-disk actual: `d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11` (matches authoritative: True)
  - Reason: prereg constant is 63 hex chars (invalid SHA-256 length; impossible to match any file); on-disk artifact independently verified genuine by the P12 source-identity record

## OBSERVED
- Host: dsw-933622-64cc845c7d-lqswc, GPU: NVIDIA A100-SXM4-80GB, torch 2.9.1+cu128, cuda 12.8
- Frozen identities all match: True
- N Gaussians: 841781; cameras: 8 ([19, 140, 219, 73, 40, 31, 255, 13])
- Timing (camera 19, 100 reps): T_screen=25.262 ms, T_global=1.014 ms (proj=0.249 vjp=0.276 opt=0.489), T_native=26.276 ms, f_global=0.039, S_max=1.040
- S_upper: K=2 1.0197, K=4 1.0298, K=8 1.0349
- R_int_energy (primary): B=2 0.9982, B=4 0.9494, B=8 0.7445
- Leave-one-out (primary, B=8): P_hold_pos=0.5133, median_C_hold=0.0000
- Diagnostic R_int_energy: B=2 0.9940, B=4 0.9400, B=8 0.8200

## EVIDENCE
- Timing segments isolated with CUDA events (warmup=10, reps=100).
- Geometry state z=(mu_2d, vech(Sigma_2d)); exact autograd Jacobian (5 VJP passes), validated vs per-Gaussian jacobian in smoke test.
- Whitening W=block_diag(Sigma^-1/2, vech-whitener); R_int=||J~ pinv(J~) d_white||^2/||d_white||^2.
- Cohort: top-4096 by accumulated |means2d.absgrad| over 8 frozen views (primary); 4096 uniform visible (diagnostic).

## HYPOTHESIS
- H1: image-rate 3D consistency is required (PSC will not promote).
- H2: multiple high-frequency projected updates can be amortized behind lower-frequency multi-view 3D sync (PSC will promote).

## TEST
- Test A (upper bound): S_upper(K)=(T_screen+T_global)/(T_screen+T_global/K).
- Test B (integrability): R_int_energy for B in {2,4,8}.
- Test C (leave-one-out): cosine consistency, P_hold_pos.

## CONCLUSION
- Mechanical verdict (owned by psc_p0_analyze.py): PSC_P0_DROP
- Analyzer stdout:
```
{
  "schema_version": "PSC-P0-v1",
  "verdict": "PSC_P0_DROP",
  "inputs": {
    "S_upper_8": 1.0349383973546105,
    "R_int_energy_B2": 0.998172402381897,
    "R_int_energy_B4": 0.9493650794029236,
    "R_int_energy_B8": 0.7444537878036499,
    "P_hold_pos_B8": 0.5133174657821655,
    "median_C_hold_B8": 0.0
  },
  "drop_rules_fired": [
    "S_upper_8 < 1.30",
    "P_hold_pos_B8 <= 0.55"
  ],
  "strong_checks": {
    "S_upper_8 >= 1.50": false,
    "R_int_energy_B8 >= 0.65": true,
    "P_hold_pos_B8 >= 0.75": false,
    "R_int_energy_B8 >= 0.50 * R_int_energy_B2": true
  },
  "fired_rules": [
    "S_upper_8 < 1.30",
    "P_hold_pos_B8 <= 0.55"
  ],
  "precedence": [
    "DROP",
    "STRONG_PROMOTE",
    "INCONCLUSIVE"
  ]
}

--STDERR--

```

Note: P0 is a falsification probe only. No production implementation is authorized unless verdict is STRONG_PROMOTE; even then, P1 is not started here.

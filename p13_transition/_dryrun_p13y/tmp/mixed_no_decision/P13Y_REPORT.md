# P13-Y ¡ª Persistent Linearization / Stale-Jacobian Oracle: report

Date: 2026-09-27. Host: EPIC-05. Decision class: **MIXED_NO_DECISION**.

Scope discipline: 8 pairs/scene is discovery only (no paper claim); no training-speedup claim; no novelty claim; gates frozen before results.

## OBSERVED

Forward Taylor linearity and stale-Jacobian backward were measured on real training revisits of the C0 arm, in the topology-frozen window 15000 <= t <= 17000 (room / bicycle / garden).

- **room**: 5/8 joint pass, 5/8 Y2 pass, 5/8 Y3 pass, 0 catastrophic, 0 invalid -> scene class **GRAY**.
- **bicycle**: 5/8 joint pass, 5/8 Y2 pass, 5/8 Y3 pass, 0 catastrophic, 0 invalid -> scene class **GRAY**.
- **garden**: 5/8 joint pass, 5/8 Y2 pass, 5/8 Y3 pass, 0 catastrophic, 0 invalid -> scene class **GRAY**.

Adapter parity gate: **PASS**. Capture neutrality: **True**. Source identity: **IDENTITY_OK**.

## EVIDENCE

### E1. Source identity (Sec.1)

| object | sha256 | expected | status |
|---|---|---|---|
| p12_trainer | 6767676767676767676767676767676767676767676767676767676767676767 | 6767676767676767676767676767676767676767676767676767676767676767 | match |
| c0_extension_so | d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2 | d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2 | match |
| c0_core_so | 9393939393939393939393939393939393939393939393939393939393939393 | 9393939393939393939393939393939393939393939393939393939393939393 | match |
| fused_ssim_cuda_so | dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd | dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd | match |

HIGS render source: `/x/gaussian_inference.py` sha256 `d5d5d5d5d5d5d5d5d5d5d5d5d5d5d5d5d5d5d5d5d5d5d5d5d5d5d5d5d5d5d5d5`; documented gsplat pin `512d366b` (dead .git link).
Capture trainer: sha256 `cacacacacacacacacacacacacacacacacacacacacacacacacacacacacacacaca` (base P12 sha `6767676767676767676767676767676767676767676767676767676767676767`), additive diff sha256 `didididididididididididididididididididididididididididididididi`.

### E2. Sequence indexing proof (Sec.3)

- L1 static: `camera_sequence.npy[i]` is the camera of training iteration `i+1` (1-based loop header, consumption index `iteration-1`); 1/1 source anchors located in the frozen P12 trainer.
- L2 regeneration: bitwise re-derivation with `random.Random(42)` pop-from-stack matches the frozen P12 30K files for all scenes: True.
- L3 empirical: in-run cam-log alignment assertion (`cam_log[i] == (i+1, camera_sequence[i])`) + parity probe per scene.
  - room: npy `/x/room.npy` dtype=int32 shape=[30000]; regeneration bitwise `True`; every-camera-once-per-cycle `True`; empirical `{'conclusion': 'ok'}`.
  - bicycle: npy `/x/bicycle.npy` dtype=int32 shape=[30000]; regeneration bitwise `True`; every-camera-once-per-cycle `True`; empirical `{'conclusion': 'ok'}`.
  - garden: npy `/x/garden.npy` dtype=int32 shape=[30000]; regeneration bitwise `True`; every-camera-once-per-cycle `True`; empirical `{'conclusion': 'ok'}`.

### E3. Revisit distribution + frozen pair plan (Sec.3)

- room: 30000 iterations, 311 cameras; adjacent revisits 8 in window [15000, 17000] (full: 5000); in-window lag min/median/max = 99/120/200.

**room frozen pair plan** (`frozen=True`, date=2026-09-27, sequence sha `abababababababab`):

| quantile | lag_target | cam | t0 | t1 | lag |
|---|---|---|---|---|---|
| 0.05 | 100 | 0 | 15000 | 15010 | 100 |
| 0.15 | 101 | 1 | 15100 | 15110 | 101 |
| 0.3 | 102 | 2 | 15200 | 15210 | 102 |
| 0.45 | 103 | 3 | 15300 | 15310 | 103 |
| 0.6 | 104 | 4 | 15400 | 15410 | 104 |
| 0.75 | 105 | 5 | 15500 | 15510 | 105 |
| 0.9 | 106 | 6 | 15600 | 15610 | 106 |
| 0.98 | 107 | 7 | 15700 | 15710 | 107 |

- bicycle: 30000 iterations, 311 cameras; adjacent revisits 8 in window [15000, 17000] (full: 5000); in-window lag min/median/max = 99/120/200.

**bicycle frozen pair plan** (`frozen=True`, date=2026-09-27, sequence sha `abababababababab`):

| quantile | lag_target | cam | t0 | t1 | lag |
|---|---|---|---|---|---|
| 0.05 | 100 | 0 | 15000 | 15010 | 100 |
| 0.15 | 101 | 1 | 15100 | 15110 | 101 |
| 0.3 | 102 | 2 | 15200 | 15210 | 102 |
| 0.45 | 103 | 3 | 15300 | 15310 | 103 |
| 0.6 | 104 | 4 | 15400 | 15410 | 104 |
| 0.75 | 105 | 5 | 15500 | 15510 | 105 |
| 0.9 | 106 | 6 | 15600 | 15610 | 106 |
| 0.98 | 107 | 7 | 15700 | 15710 | 107 |

- garden: 30000 iterations, 311 cameras; adjacent revisits 8 in window [15000, 17000] (full: 5000); in-window lag min/median/max = 99/120/200.

**garden frozen pair plan** (`frozen=True`, date=2026-09-27, sequence sha `abababababababab`):

| quantile | lag_target | cam | t0 | t1 | lag |
|---|---|---|---|---|---|
| 0.05 | 100 | 0 | 15000 | 15010 | 100 |
| 0.15 | 101 | 1 | 15100 | 15110 | 101 |
| 0.3 | 102 | 2 | 15200 | 15210 | 102 |
| 0.45 | 103 | 3 | 15300 | 15310 | 103 |
| 0.6 | 104 | 4 | 15400 | 15410 | 104 |
| 0.75 | 105 | 5 | 15500 | 15510 | 105 |
| 0.9 | 106 | 6 | 15600 | 15610 | 106 |
| 0.98 | 107 | 7 | 15700 | 15710 | 107 |

### E4. Capture neutrality (Sec.4)

- CPU/NumPy/Torch-CPU/Torch-CUDA RNG saved+restored around every capture (`rng_restored_exactly` asserted in-run per capture).
- Trio runs: {"off1": {"dir": "/off1"}}
- camera_sequence identical in all three: True; final params OFF1==OFF2: True, ON==OFF1: True.
- noise floor (OFF1 vs OFF2): {"n_loss_differing": 3}
- capture effect (ON vs OFF1): {"n_loss_differing": 2}
- Criterion pass: **True** (r) checks={"a": true}.
- Full capture replay vs P12 30K prefix: n_rows=171, n_loss_differing=4, max_abs_loss_diff=1e-06, first_differing_iter=14, n_gs_equal_all=True, camera_seq_match=True.

### E5. Adapter parity (Sec.5)

- room: status=PASS, n_probes=1, max|I_A-I_B|=0, max loss rel err=0
- bicycle: status=PASS, n_probes=1, max|I_A-I_B|=0, max loss rel err=0
- garden: status=PASS, n_probes=1, max|I_A-I_B|=0, max loss rel err=0

### E6. Y1 graph stability (Sec.6, descriptive only)

- room medians:

| visible_jaccard | membership_jaccard | order_inversion_rate | unchanged_full_order_frac |
|---|---|---|---|
| 0.99 | 0.98 | 0.01 | 0.9 |

- bicycle medians:

| visible_jaccard | membership_jaccard | order_inversion_rate | unchanged_full_order_frac |
|---|---|---|---|
| 0.99 | 0.98 | 0.01 | 0.9 |

- garden medians:

| visible_jaccard | membership_jaccard | order_inversion_rate | unchanged_full_order_frac |
|---|---|---|---|
| 0.99 | 0.98 | 0.01 | 0.9 |

### E7. Y2 forward linearity (Sec.7)

- room medians:

| E_delta | E_I | PSNR(I_lin,I1) | L_rel_err | h_consistency_max_rel |
|---|---|---|---|---|
| 0.2 | 0.02 | 40 | 0.001 | 0.01 |

- bicycle medians:

| E_delta | E_I | PSNR(I_lin,I1) | L_rel_err | h_consistency_max_rel |
|---|---|---|---|---|
| 0.2 | 0.02 | 40 | 0.001 | 0.01 |

- garden medians:

| E_delta | E_I | PSNR(I_lin,I1) | L_rel_err | h_consistency_max_rel |
|---|---|---|---|---|
| 0.2 | 0.02 | 40 | 0.001 | 0.01 |

### E8. Y3 stale-Jacobian backward (Sec.8)

- room medians:

| cos_means | E_rel_means | cos_opacity | cos_scales | cos_quats | cos_sh0 | cos_shN |
|---|---|---|---|---|---|---|
| 0.999 | 0.01 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 |

- bicycle medians:

| cos_means | E_rel_means | cos_opacity | cos_scales | cos_quats | cos_sh0 | cos_shN |
|---|---|---|---|---|---|---|
| 0.999 | 0.01 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 |

- garden medians:

| cos_means | E_rel_means | cos_opacity | cos_scales | cos_quats | cos_sh0 | cos_shN |
|---|---|---|---|---|---|---|
| 0.999 | 0.01 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 |

### E9. Per-pair table

**room**

| i | cam | t0 | t1 | lag | E_delta | L_rel | cos_mu | E_rel_mu | cos_opa | Y1_visJ | v |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 15000 | 15010 | 100 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | J |
| 1 | 1 | 15100 | 15110 | 101 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | J |
| 2 | 2 | 15200 | 15210 | 102 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | J |
| 3 | 3 | 15300 | 15310 | 103 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | J |
| 4 | 4 | 15400 | 15410 | 104 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | J |
| 5 | 5 | 15500 | 15510 | 105 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | f |
| 6 | 6 | 15600 | 15610 | 106 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | f |
| 7 | 7 | 15700 | 15710 | 107 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | f |

**bicycle**

| i | cam | t0 | t1 | lag | E_delta | L_rel | cos_mu | E_rel_mu | cos_opa | Y1_visJ | v |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 15000 | 15010 | 100 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | J |
| 1 | 1 | 15100 | 15110 | 101 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | J |
| 2 | 2 | 15200 | 15210 | 102 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | J |
| 3 | 3 | 15300 | 15310 | 103 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | J |
| 4 | 4 | 15400 | 15410 | 104 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | J |
| 5 | 5 | 15500 | 15510 | 105 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | f |
| 6 | 6 | 15600 | 15610 | 106 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | f |
| 7 | 7 | 15700 | 15710 | 107 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | f |

**garden**

| i | cam | t0 | t1 | lag | E_delta | L_rel | cos_mu | E_rel_mu | cos_opa | Y1_visJ | v |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 15000 | 15010 | 100 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | J |
| 1 | 1 | 15100 | 15110 | 101 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | J |
| 2 | 2 | 15200 | 15210 | 102 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | J |
| 3 | 3 | 15300 | 15310 | 103 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | J |
| 4 | 4 | 15400 | 15410 | 104 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | J |
| 5 | 5 | 15500 | 15510 | 105 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | f |
| 6 | 6 | 15600 | 15610 | 106 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | f |
| 7 | 7 | 15700 | 15710 | 107 | 0.2 | 0.001 | 0.999 | 0.01 | 0.999 | 0.99 | f |

### E10. Contamination ledger (Sec.12)

- units logged: 1; [('dry_start', 0)]
- policy: one clean physical GPU cohort per compared batch; GPU index/UUID + compute processes logged before/after every unit; contaminated units retained verbatim

### E11. Manifest

- 43 files, 0.00 GB under `C:\Users\36570\3dgs-renderer-benchmark\p13_transition\_dryrun_p13y\tmp\mixed_no_decision` (sha256 per file in MANIFEST_P13Y.json).

## HYPOTHESIS

H1 (forward): for a fixed real training camera c and adjacent true revisits t0->t1 in the post-densification window, the render is locally affine in the raw parameter vector: `R_c(theta1) ~ R_c(theta0) + J_c(theta0)(theta1-theta0)` with E_delta <= 0.10.

H2 (backward): with one frozen adjoint lambda1 = dL/dI at t1, the stale Jacobian transpose delivers the same parameter gradient: `J_c(theta0)^T lambda1 ~ J_c(theta1)^T lambda1`, cos >= 0.995 on means+opacity (secondary groups cos >= 0.990), E_rel <= 0.05 (0.10).

H3 (compositional): if H1+H2 hold with a high joint-pass rate across scenes, a persistent linearization oracle could skip relinearization for a bounded reuse lifetime; otherwise the mechanism dies and the program moves to Spectral / Compressed Adjoint without rescue.

H1/H2 as stated are the operands of the frozen gates; no gate was edited after any result.

## TEST

- Sec.7 Y2: d = theta1-theta0; central FD at h=0.01 (primary), crosschecked at h=0.005/0.02 with NO post-hoc h selection; I_lin = I0 + J0 d; primary E_delta = ||I_lin-I1||/||I1-I0||; Y2 valid iff ||I1-I0||/||I1|| >= 1e-6 and h-consistency <= 0.05; Y2 PASS iff E_delta <= 0.10 and |L(I_lin)-L(I1)|/|L(I1)| <= 0.005.
- Sec.8 Y3: at theta1, lambda1 = dL(I1,GT)/dI1 (fused training loss); g_true = J(theta1)^T lambda1, g_stale = J(theta0)^T lambda1 (same lambda1, fresh graphs); per-group cosine / relative L2 / norm ratio / top-10%-magnitude sign agreement; primary means+opacity cos >= 0.995 and E_rel <= 0.05, secondary scales/quats/sh0/shN cos >= 0.990 and E_rel <= 0.10; a two-step-vs-direct gradient check validates the split path.
- Sec.9 scene gate and Sec.10 global classes were evaluated exactly as frozen (no post-hoc edits).
- Y1 (Sec.6) is descriptive only and did not decide linearization; Y2/Y3 ran on ALL selected pairs.

## CONCLUSION

**MIXED_NO_DECISION**

- scene classes: {"room": "GRAY", "bicycle": "GRAY", "garden": "GRAY"}
- adapter parity gate: PASS
- capture neutrality pass: True
- next step: move to Spectral / Compressed Adjoint; do NOT rescue persistent linearization

Per Sec.13 the program stops here for this mechanism and moves to Spectral / Compressed Adjoint; persistent linearization is NOT rescued.

### Failures / anomalies preserved

- invalid pairs: 0; catastrophic pairs: 0 (per-pair JSONs and tracebacks kept verbatim under pair_results/).
- contaminated units (if any) are retained verbatim in logs/P13Y_CONTAMINATION.log and P13Y_CONTAMINATION.json.
- no production persistent CUDA cache was written; no 30K full training was started; no external speedup/novelty claim is made; no git commit.

# P13-Y — Persistent Linearization / Stale-Jacobian Oracle: report

Date: 2026-09-27. Host: EPIC-05. Decision class: **DROP_PERSISTENT_LINEARIZATION**.

Scope discipline: 8 pairs/scene is discovery only (no paper claim); no training-speedup claim; no novelty claim; gates frozen before results.

## OBSERVED

Forward Taylor linearity and stale-Jacobian backward were measured on real training revisits of the C0 arm, in the topology-frozen window 15000 <= t <= 17000 (room / bicycle / garden).

- **room**: 0/8 joint pass, 0/8 Y2 pass, 0/8 Y3 pass, 8 catastrophic, 0 invalid -> scene class **DROP_TRIGGER**.
- **bicycle**: 0/8 joint pass, 0/8 Y2 pass, 0/8 Y3 pass, 8 catastrophic, 0 invalid -> scene class **DROP_TRIGGER**.
- **garden**: 0/8 joint pass, 0/8 Y2 pass, 0/8 Y3 pass, 8 catastrophic, 0 invalid -> scene class **DROP_TRIGGER**.

Adapter parity gate: **PASS**. Capture neutrality: **False**. Source identity: **IDENTITY_OK**.

## EVIDENCE

### E1. Source identity (Sec.1)

| object | sha256 | expected | status |
|---|---|---|---|
| p12_trainer | 675b0267f3a1a6a0fa63fadb37c8536481c1c7a6de37a4b7f6ffa98a449050c5 | 675b0267f3a1a6a0fa63fadb37c8536481c1c7a6de37a4b7f6ffa98a449050c5 | match |
| c0_extension_so | d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11 | d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11 | match |
| c0_core_so | 935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e | 935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e | match |
| fused_ssim_cuda_so | dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb | dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb | match |

HIGS render source: `/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/experimental/render/functional/gaussian_inference.py` sha256 `d599aae3c1864993934d65646bfcb33e07e6046e389f028230752200df0c3a8a`; documented gsplat pin `512d366b67073d77ca099ede742683c165dfc23b` (worktree .git link dead; file hash is the operative identity).
Capture trainer: sha256 `d9781be5ca224597672f6526ded90dc6139adaace3dfff09cab2a7d1ea33e185` (base P12 sha `675b0267f3a1a6a0fa63fadb37c8536481c1c7a6de37a4b7f6ffa98a449050c5`), additive diff sha256 `7f5f23b56c25639f3a6d25f06ffbfebc3bb1cac2fb782d9be924833f2bac6f52`.

### E2. Sequence indexing proof (Sec.3)

- L1 static: `camera_sequence.npy[i]` is the camera of training iteration `i+1` (1-based loop header, consumption index `iteration-1`); 8/8 source anchors located in the frozen P12 trainer.
- L2 regeneration: bitwise re-derivation with `random.Random(42)` pop-from-stack matches the frozen P12 30K files for all scenes: True.
- L3 empirical: in-run cam-log alignment assertion (`cam_log[i] == (i+1, camera_sequence[i])`) + parity probe per scene.
  - room: npy `/root/epic05_p12/runs/p12a/p12a_c0f_s42_room/camera_sequence.npy` dtype=int32 shape=[30000]; regeneration bitwise `True`; every-camera-once-per-cycle `True`; empirical `{'cam_log': 'runs/y_capture_room/cam_log.json', 'n_logged': 17000, 'run_iterations': 17000, 'run_camera_sequence_sha256': 'bdbb7db230a61d545a47848e34671f3642b84f74f6f011c09c057909a2d0279d', 'alignment_rule': 'cam_log[i][0] == i+1 and cam_log[i][1] == camera_sequence[i] for every logged iteration', 'alignment_holds_in_run': True, 'cam_log_sha256': 'aef7b4eb72cfdf5a340cf067acd0ec6d33d393f3354a13aaa104db0836fb43ba', 'run_npy': 'runs/y_capture_room/camera_sequence.npy', 'run_npy_sha256': 'bdbb7db230a61d545a47848e34671f3642b84f74f6f011c09c057909a2d0279d', 'run_npy_len': 17000, 'run_npy_matches_logged_cameras': True, 'p12_prefix_len_compared': 17000, 'run_npy_is_p12_prefix': True, 'conclusion': "camera_sequence.npy[i] is the camera of training iteration i+1 (1-based); the replay's own sequence is the frozen P12 30K prefix"}`.
  - bicycle: npy `/root/epic05_p12/runs/p12a/p12a_c0f_s42_bicycle/camera_sequence.npy` dtype=int32 shape=[30000]; regeneration bitwise `True`; every-camera-once-per-cycle `True`; empirical `{'cam_log': 'runs/y_capture_bicycle/cam_log.json', 'n_logged': 17000, 'run_iterations': 17000, 'run_camera_sequence_sha256': '750e72c97d90cf04d00d60c028e6f7818d49aca3b88779b6559d31c7e82d41a2', 'alignment_rule': 'cam_log[i][0] == i+1 and cam_log[i][1] == camera_sequence[i] for every logged iteration', 'alignment_holds_in_run': True, 'cam_log_sha256': 'bece4b1335936bc2449009e97d82477d2c60d03c19414a3dc45e22a79e57397c', 'run_npy': 'runs/y_capture_bicycle/camera_sequence.npy', 'run_npy_sha256': '750e72c97d90cf04d00d60c028e6f7818d49aca3b88779b6559d31c7e82d41a2', 'run_npy_len': 17000, 'run_npy_matches_logged_cameras': True, 'p12_prefix_len_compared': 17000, 'run_npy_is_p12_prefix': True, 'conclusion': "camera_sequence.npy[i] is the camera of training iteration i+1 (1-based); the replay's own sequence is the frozen P12 30K prefix"}`.
  - garden: npy `/root/epic05_p12/runs/p12a/p12a_c0f_s42_garden/camera_sequence.npy` dtype=int32 shape=[30000]; regeneration bitwise `True`; every-camera-once-per-cycle `True`; empirical `{'cam_log': 'runs/y_capture_garden/cam_log.json', 'n_logged': 17000, 'run_iterations': 17000, 'run_camera_sequence_sha256': '888c4f26280bfc226d264ecb494b6ed741acf5a67da4480a010428f78b50745a', 'alignment_rule': 'cam_log[i][0] == i+1 and cam_log[i][1] == camera_sequence[i] for every logged iteration', 'alignment_holds_in_run': True, 'cam_log_sha256': '09151075322ebf3e33aa54ce4075c9a7af9c97db3d5e9d94659f0ad193c5f209', 'run_npy': 'runs/y_capture_garden/camera_sequence.npy', 'run_npy_sha256': '888c4f26280bfc226d264ecb494b6ed741acf5a67da4480a010428f78b50745a', 'run_npy_len': 17000, 'run_npy_matches_logged_cameras': True, 'p12_prefix_len_compared': 17000, 'run_npy_is_p12_prefix': True, 'conclusion': "camera_sequence.npy[i] is the camera of training iteration i+1 (1-based); the replay's own sequence is the frozen P12 30K prefix"}`.

### E3. Revisit distribution + frozen pair plan (Sec.3)

- room: 30000 iterations, 311 cameras; adjacent revisits 1690 in window [15000, 17000] (full: 29689); in-window lag min/median/max = 7/297/605.

**room frozen pair plan** (`frozen=True`, date=2026-09-27, sequence sha `a80127c542dc2859`):

| quantile | lag_target | cam | t0 | t1 | lag |
|---|---|---|---|---|---|
| 0.05 | 91.45 | 84 | 16104 | 16195 | 91 |
| 0.15 | 162 | 6 | 16420 | 16582 | 162 |
| 0.3 | 227 | 22 | 15073 | 15300 | 227 |
| 0.45 | 282 | 31 | 15238 | 15520 | 282 |
| 0.6 | 330 | 69 | 16096 | 16426 | 330 |
| 0.75 | 390 | 184 | 15968 | 16358 | 390 |
| 0.9 | 466 | 27 | 15913 | 16379 | 466 |
| 0.98 | 544.2 | 113 | 15620 | 16164 | 544 |

- bicycle: 30000 iterations, 194 cameras; adjacent revisits 1807 in window [15000, 17000] (full: 29806); in-window lag min/median/max = 7/190/380.

**bicycle frozen pair plan** (`frozen=True`, date=2026-09-27, sequence sha `8debe1c2244aca09`):

| quantile | lag_target | cam | t0 | t1 | lag |
|---|---|---|---|---|---|
| 0.05 | 59.3 | 42 | 15707 | 15766 | 59 |
| 0.15 | 103 | 26 | 16216 | 16319 | 103 |
| 0.3 | 145.8 | 4 | 15688 | 15834 | 146 |
| 0.45 | 179 | 42 | 15766 | 15945 | 179 |
| 0.6 | 211 | 2 | 15653 | 15864 | 211 |
| 0.75 | 247 | 4 | 16722 | 16969 | 247 |
| 0.9 | 293 | 28 | 15186 | 15479 | 293 |
| 0.98 | 337 | 46 | 16492 | 16829 | 337 |

- garden: 30000 iterations, 185 cameras; adjacent revisits 1816 in window [15000, 17000] (full: 29815); in-window lag min/median/max = 1/185/363.

**garden frozen pair plan** (`frozen=True`, date=2026-09-27, sequence sha `63ad2de2256d0905`):

| quantile | lag_target | cam | t0 | t1 | lag |
|---|---|---|---|---|---|
| 0.05 | 59 | 17 | 15506 | 15565 | 59 |
| 0.15 | 103.2 | 11 | 16398 | 16501 | 103 |
| 0.3 | 144.5 | 3 | 16089 | 16234 | 145 |
| 0.45 | 176 | 4 | 16217 | 16393 | 176 |
| 0.6 | 204 | 9 | 15005 | 15209 | 204 |
| 0.75 | 234 | 35 | 15599 | 15833 | 234 |
| 0.9 | 282 | 8 | 15021 | 15303 | 282 |
| 0.98 | 323.7 | 82 | 15377 | 15701 | 324 |

### E4. Capture neutrality (Sec.4)

- CPU/NumPy/Torch-CPU/Torch-CUDA RNG saved+restored around every capture (`rng_restored_exactly` asserted in-run per capture).
- Trio runs: {"off1": {"dir": "runs/y_neutral_room_off1", "iters": 3500, "final_N": 332986, "camera_sequence_sha256": "0772cbd0beb964723e59d453fbaa3e35dc6ff8001b6f1694b0fdfc44d223fcac", "final_model_param_sha256": "15242e643e02a1e6f390dc0e47311a63bdb40579d99fd414c908a49fb7765b60"}, "on": {"dir": "runs/y_neutral_room_on", "iters": 3500, "final_N": 334488, "camera_sequence_sha256": "0772cbd0beb964723e59d453fbaa3e35dc6ff8001b6f1694b0fdfc44d223fcac", "final_model_param_sha256": "627178f262f19908e01249357585ddac5430789428e7bf52f700e5feadfedcb8"}, "off2": {"dir": "runs/y_neutral_room_off2", "iters": 3500, "final_N": 332163, "camera_sequence_sha256": "0772cbd0beb964723e59d453fbaa3e35dc6ff8001b6f1694b0fdfc44d223fcac", "final_model_param_sha256": "91edba0ede4edf0ba69fc84d242d852aeb91daa4fcd1c01cb44ac11b5001f227"}}
- camera_sequence identical in all three: True; final params OFF1==OFF2: False, ON==OFF1: False.
- noise floor (OFF1 vs OFF2): {"n_common": 3500, "n_loss_exact_equal": 87, "n_loss_differing": 3413, "max_abs_loss_diff": 0.24244987219572067, "first_differing_iter": 14, "first_differing_iter_ge10": 14, "final_n_gs_equal": false}
- capture effect (ON vs OFF1): {"n_common": 3500, "n_loss_exact_equal": 81, "n_loss_differing": 3419, "max_abs_loss_diff": 0.1623624563217163, "first_differing_iter": 15, "first_differing_iter_ge10": 15, "final_n_gs_equal": false}
- Criterion pass: **False** (capture effect (ON vs OFF1) must not exceed the intrinsic cross-process noise floor (OFF1 vs OFF2) per metric) checks={"n_loss_differing": false, "max_abs_loss_diff": true, "first_differing_iter": false}.

#### E4b. Neutrality diagnosis (post-hoc, appended after the frozen verdict)

The frozen criterion is a pointwise single-sample comparison against one OFF replay; it was evaluated before this diagnosis and is NOT edited here. What the diagnosis adds:

- Cross-process float nondeterminism in the frozen C0 recipe seeds at iteration ~14 with NO instrumentation at all (OFF1 vs OFF2 diverge at iter 14); it was already documented in the P13-X preflight as C_vs_D ~ D_vs_D2. The same seed iteration is therefore present in every compared pair, so the trio comparison is a chaotic-amplification comparison, not a deterministic reproduction test.
- Short trio (400 iters): noise floor max|dL|=1.769e-05 (first diff iter 14) vs capture effect max|dL|=0.0007627 (first diff iter 23).
- Five-OFF ensemble vs one ON (400 iters): OFF-OFF max|dL| spans 5.879e-06..2.727e-05 (median 1.769e-05); ON-OFF spans 0.0007624..0.0007715 -> ON lies OUTSIDE the OFF-OFF envelope (envelope_ok=False), i.e. beyond replay-to-replay spread.
  - end-state psnr: OFF 17.407255..17.407311, ON 17.407428 (inside envelope False).
  - end-state ssim: OFF 0.656730493..0.656731715, ON 0.656709296 (inside envelope False).
  - end-state lpips: OFF 0.623387342..0.623391454, ON 0.623400249 (inside envelope False).
- OUTCOME (recorded, no gate): the short-horizon (400-iter) instrumented run is a distributional outlier -- every ON-OFF max|dL| is larger than every OFF-OFF max|dL| (0.0007715 vs 2.727e-05, ~28x), while the ON end-state quality deviations are in the 1e-4 dB / 3e-5 relative range. At the longer horizon (3500 iters) ON-OFF max|dL|=0.1624 is BELOW the OFF-OFF max 0.2424, so the outlier pattern does not persist at 3500 iterations. Both observations are recorded as-is; neither alters the frozen criterion verdict (FAIL) nor the sec.9/sec.10 class.
- Reading (recorded, not a gate): If every ON-OFF value lies inside the OFF-OFF envelope (and the ON end-state quality lies inside the OFF ensemble spread), the capture side channel shows no deviation beyond the intrinsic cross-process divergence of the C0 recipe.
- Consequence for this phase: Y2/Y3 compare theta0 and theta1 taken from the SAME run along the SAME trajectory, so the oracle measurement itself is internally valid; but the capture side channel is NOT proven pointwise-neutral, so any future design that assumes provable capture neutrality must re-establish it (e.g. state capture from a deterministic replay checkpoint, or an out-of-process re-render) before that assumption is used.

### E5. Adapter parity (Sec.5)

- room: status=PASS, n_probes=16, max|I_A-I_B|=0, max loss rel err=0
- bicycle: status=PASS, n_probes=15, max|I_A-I_B|=0, max loss rel err=0
- garden: status=PASS, n_probes=16, max|I_A-I_B|=0, max loss rel err=0

### E6. Y1 graph stability (Sec.6, descriptive only)

- room medians:

| visible_jaccard | membership_jaccard | order_inversion_rate | unchanged_full_order_frac |
|---|---|---|---|
| 0.9965 | 0.9313 | 0.001536 | 0.0005515 |

- bicycle medians:

| visible_jaccard | membership_jaccard | order_inversion_rate | unchanged_full_order_frac |
|---|---|---|---|
| 0.9988 | 0.9343 | 0.0007072 | 0.0001838 |

- garden medians:

| visible_jaccard | membership_jaccard | order_inversion_rate | unchanged_full_order_frac |
|---|---|---|---|
| 0.9984 | 0.9464 | 0.0008362 | 0 |

### E7. Y2 forward linearity (Sec.7)

- room medians:

| E_delta | E_I | PSNR(I_lin,I1) | L_rel_err | h_consistency_max_rel |
|---|---|---|---|---|
| 0.9372 | 0.04693 | 35.68 | 0.5808 | 0.6785 |

- bicycle medians:

| E_delta | E_I | PSNR(I_lin,I1) | L_rel_err | h_consistency_max_rel |
|---|---|---|---|---|
| 1.088 | 0.09859 | 28.58 | 0.2234 | 0.6807 |

- garden medians:

| E_delta | E_I | PSNR(I_lin,I1) | L_rel_err | h_consistency_max_rel |
|---|---|---|---|---|
| 1.046 | 0.06377 | 31.32 | 0.1492 | 0.6964 |

### E8. Y3 stale-Jacobian backward (Sec.8)

- room medians:

| cos_means | E_rel_means | cos_opacity | cos_scales | cos_quats | cos_sh0 | cos_shN |
|---|---|---|---|---|---|---|
| 0.8647 | 0.5058 | 0.9587 | 0.9449 | 0.7702 | 0.9789 | 0.9789 |

- bicycle medians:

| cos_means | E_rel_means | cos_opacity | cos_scales | cos_quats | cos_sh0 | cos_shN |
|---|---|---|---|---|---|---|
| 0.8425 | 0.5496 | 0.9493 | 0.8899 | 0.5677 | 0.9594 | 0.9594 |

- garden medians:

| cos_means | E_rel_means | cos_opacity | cos_scales | cos_quats | cos_sh0 | cos_shN |
|---|---|---|---|---|---|---|
| 0.9412 | 0.3384 | 0.9296 | 0.9063 | 0.7545 | 0.9723 | 0.9723 |

### E9. Per-pair table

**room**

| i | cam | t0 | t1 | lag | E_delta | L_rel | cos_mu | E_rel_mu | cos_opa | Y1_visJ | v |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 84 | 16104 | 16195 | 91 | 0.3655 | 0.2998 | 0.8229 | 0.5799 | 0.9143 | 0.9981 | f |
| 1 | 6 | 16420 | 16582 | 162 | 0.8832 | 0.4665 | 0.8593 | 0.5142 | 0.9642 | 0.9978 | f |
| 2 | 22 | 15073 | 15300 | 227 | 0.9635 | 0.5157 | 0.8701 | 0.4973 | 0.9531 | 0.9971 | f |
| 3 | 31 | 15238 | 15520 | 282 | 0.912 | 0.2961 | 0.9253 | 0.3817 | 0.9777 | 0.9958 | f |
| 4 | 69 | 16096 | 16426 | 330 | 0.9624 | 0.6516 | 0.9108 | 0.4206 | 0.9511 | 0.996 | f |
| 5 | 184 | 15968 | 16358 | 390 | 1.132 | 0.6661 | 0.7906 | 0.6401 | 0.9448 | 0.9971 | f |
| 6 | 27 | 15913 | 16379 | 466 | 0.846 | 0.6459 | 0.7727 | 0.6567 | 0.9753 | 0.9942 | f |
| 7 | 113 | 15620 | 16164 | 544 | 0.9896 | 0.7698 | 0.8823 | 0.4738 | 0.9826 | 0.9958 | f |

**bicycle**

| i | cam | t0 | t1 | lag | E_delta | L_rel | cos_mu | E_rel_mu | cos_opa | Y1_visJ | v |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 42 | 15707 | 15766 | 59 | 0.8936 | 0.2397 | 0.8259 | 0.5693 | 0.9574 | 0.999 | f |
| 1 | 26 | 16216 | 16319 | 103 | 1.017 | 0.2284 | 0.8697 | 0.4936 | 0.9927 | 0.9989 | f |
| 2 | 4 | 15688 | 15834 | 146 | 1.099 | 0.1695 | 0.7022 | 0.7513 | 0.9127 | 0.999 | f |
| 3 | 42 | 15766 | 15945 | 179 | 1.117 | 0.274 | 0.859 | 0.5298 | 0.9535 | 0.9986 | f |
| 4 | 2 | 15653 | 15864 | 211 | 1.192 | 0.2184 | 0.8833 | 0.4953 | 0.9634 | 0.9988 | f |
| 5 | 4 | 16722 | 16969 | 247 | 1.197 | 0.1954 | 0.7549 | 0.6857 | 0.945 | 0.9985 | f |
| 6 | 28 | 15186 | 15479 | 293 | 1.077 | 0.2334 | 0.7751 | 0.6407 | 0.9157 | 0.999 | f |
| 7 | 46 | 16492 | 16829 | 337 | 0.7649 | 0.1022 | 0.8798 | 0.4809 | 0.8983 | 0.9978 | f |

**garden**

| i | cam | t0 | t1 | lag | E_delta | L_rel | cos_mu | E_rel_mu | cos_opa | Y1_visJ | v |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 17 | 15506 | 15565 | 59 | 0.1815 | 0.03153 | 0.2935 | 1.218 | 0.6213 | 0.9985 | f |
| 1 | 11 | 16398 | 16501 | 103 | 0.8388 | 0.1678 | 0.9467 | 0.3223 | 0.9269 | 0.9988 | f |
| 2 | 3 | 16089 | 16234 | 145 | 1.034 | 0.1351 | 0.9415 | 0.337 | 0.9475 | 0.9985 | f |
| 3 | 4 | 16217 | 16393 | 176 | 0.9664 | 0.1469 | 0.8645 | 0.5119 | 0.9322 | 0.9982 | f |
| 4 | 9 | 15005 | 15209 | 204 | 1.247 | 0.144 | 0.967 | 0.255 | 0.9673 | 0.9983 | f |
| 5 | 35 | 15599 | 15833 | 234 | 1.101 | 0.189 | 0.8788 | 0.4864 | 0.9344 | 0.998 | f |
| 6 | 8 | 15021 | 15303 | 282 | 1.087 | 0.2289 | 0.9409 | 0.3399 | 0.8309 | 0.9986 | f |
| 7 | 82 | 15377 | 15701 | 324 | 1.057 | 0.1515 | 0.9614 | 0.2758 | 0.9152 | 0.9983 | f |

### E10. Contamination ledger (Sec.12)

- units logged: 26; [('y0a_identity', 0), ('y0b_sequence', 0), ('y_smoke_room', 0), ('y_parity_smoke', 1), ('y_parity_smoke', 0), ('y_neutral_off1', 0), ('y_neutral_on', 0), ('y_neutral_off2', 0), ('y_neutral_s_off1', 0), ('y_neutral_s_on', 0), ('y_neutral_s_off2', 0), ('y_neutral_s_off3', 0), ('y_neutral_s_off4', 0), ('y_neutral_s_off5', 0), ('y_parity_trio', 1), ('y_parity_trio', 1), ('y_parity_trio', 0), ('y_capture_room', 0), ('y_capture_bicycle', 0), ('y_capture_garden', 0), ('y_parity_room', 0), ('y_oracle_room', 0), ('y_parity_bicycle', 0), ('y_oracle_bicycle', 0), ('y_parity_garden', 0), ('y_oracle_garden', 0)]
- policy: one clean physical GPU cohort per compared batch; GPU index/UUID + compute processes logged before/after every unit; contaminated units retained verbatim

### E11. Manifest

- 555 files, 30.12 GB under `/root/epic05_p13y` (sha256 per file in MANIFEST_P13Y.json).

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

**DROP_PERSISTENT_LINEARIZATION**

- scene classes: {"room": "DROP_TRIGGER", "bicycle": "DROP_TRIGGER", "garden": "DROP_TRIGGER"}
- adapter parity gate: PASS
- capture neutrality pass: False
- next step: move to Spectral / Compressed Adjoint; do NOT rescue persistent linearization

Per Sec.13 the program stops here for this mechanism and moves to Spectral / Compressed Adjoint; persistent linearization is NOT rescued.

### Failures / anomalies preserved

- invalid pairs: 0; catastrophic pairs: 24 (per-pair JSONs and tracebacks kept verbatim under pair_results/).
- contaminated units (if any) are retained verbatim in logs/P13Y_CONTAMINATION.log and P13Y_CONTAMINATION.json.
- no production persistent CUDA cache was written; no 30K full training was started; no external speedup/novelty claim is made; no git commit.

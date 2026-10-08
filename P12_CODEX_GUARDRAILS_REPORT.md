# P12 Codex guardrails report

Date: 2026-09-26

## Scope and assumptions

These scripts validate the P12 protocol without launching training or modifying renderer CUDA. The P12-A arms are `GMAIN_FUSED` and `C0_FUSED`; P12-B uses `A0_FUSED` and `A1_FUSED`. The canonical scene order is the 13-scene list in the frozen P12 prompt. A recorded `rasterization_mode` is `classic` for current gsplat and `higs_dynamic` for the C0 substrate; C0-family backward mode is separately `higs_native`.

The local P8/P11 artifacts pin the GMAIN `.so` at `8ca8ec1e…58c811`, C0/A0/A1 `.so` at `9baf8655…ed95b91`, fused-SSIM `.so` at `dd6a4c79…d8ebb`, current gsplat source commit at `512d366b…5dfc23b`, and fused-SSIM source commit at `a7c48d6d…418f38`. The local artifacts do not contain a cryptographic C0 source-tree hash. That expected hash and the historical P8 source snapshot must be supplied from the frozen publication environment; the guards fail if the identity is missing.

## Run guard

`p12_run_guard.py` accepts `--config` and `--output-dir`. It measures git commits, source cleanliness, renderer and fused-SSIM binaries, and the local fused-SSIM Git tree hash. The trainer's process-start attestation supplies the active scene, arm, loss and renderer arguments, source paths, and switches. Call `guard_run(config, output_dir)` before renderer import or any training step, or run the CLI as a mandatory launch preflight. A failed invariant writes `RUN_GUARD_FAIL.json`, prints the exact violations, and exits nonzero.

Required config fields: `arm`, `runtime_arm`, `scene`, `expected_scene`, `padding`, `eps2d`, `seed`, `loss_weights` (`{"l1":0.8,"ssim":0.2}`), `absgrad`, `absgrad_threshold`, `rasterization_mode`, `iterations`, `expected_repo_sha`, `repo_dir`, `renderer_repo`, `renderer_so`, `fused_ssim_so`, `switches`, and `c0_custom_backward` for GMAIN. C0-family arms also require `backward_mode`, `expected_renderer_commit`, `renderer_source_manifest`, and `expected_renderer_source_sha256`. Pass `fused_ssim_repo` when the source checkout is local. Switches are checked against both the actual process environment and the attestation. `renderer_source_manifest` must be a frozen, independently established source snapshot outside the renderer checkout; its SHA256 must match the historical expected value.

The guard cannot observe Python call arguments by itself. The calling trainer must pass its actual effective arguments, not intended defaults. Source equality also requires the diff guard below, since an expected C0 source hash is not present in the local P8 JSON.

## Pair manifest

`p12_pair_manifest.py --campaign P12-A|P12-B --runs RUNS.json --order-file ORDER.json --output MANIFEST.json` consumes a `{"runs":[...]}` file with one record per attempt and a preregistered scene-to-arm-order JSON map. It requires 26 clean runs, the canonical alternating order, one physical GPU UUID and host per pair, nonoverlapping timestamps, identical common recipe fields, and the fixed fused-SSIM identity. Each run records `scene`, `arm`, `gpu_uuid`, `hostname`, timezone-aware `start`/`end`, `repo_sha`, `renderer_so_sha256`, `fused_ssim_so_sha256`, `fused_ssim_source_commit`, `command`, `status`, `contaminated` (explicit `false`), `rasterization_mode`, and `recipe`. The recipe records seed, eps2d, padding, loss weights, absgrad, threshold, iterations, and hashes of camera sequence, dataset, optimizer, schedule, and evaluation policy. The output retains every pair's run records and violations.

The manifest can verify only runs present in the supplied raw record. External GPU jobs require separate contamination monitoring and cannot be inferred from timestamps alone.

## Statistics checker

`p12_stats_check.py --input RAW.json --output STATS.json` consumes `{"campaign":"P12-A|P12-B","runs":[...],"reported":{"gm":...,"wall_reduction":...}}`. Each clean run has `scene`, `arm`, `wall_time_s`, `final_n`, `psnr`, `ssim`, `lpips`, `status:"clean"`, and `contaminated:false`. Rejected attempts remain in the raw file but are excluded. All 13 scenes still need a complete clean pair; duplicate clean runs fail.

Per scene `ratio = wall_base / wall_test`. `GM = exp(mean(log(ratio)))` and wall reduction is `1 - 1/GM`. The CI is always a two-sided Student t interval on the 13 paired log ratios, with sample standard deviation, 12 degrees of freedom, and fixed `t(0.975)=2.1788128296634177`; there is no method fallback. The checker reports the log interval and exponentiated ratio interval, median/min/max ratio, positive count, quality mean and median deltas, per-scene final-N ratio `test/base`, and Pearson correlation of log wall and log N ratios when variance is nonzero. Quality deltas follow the frozen prompt: GMAIN minus C0 for P12-A and A1 minus A0 for P12-B. Reported GM and reduction must match the recomputation within `5e-5` in fractional units, which accommodates four-decimal rounding while rejecting `GM - 1` as a reduction.

Regression: `GM=1.0979` maps to approximately **8.92%** wall-time reduction, not 9.79%.

## Diff guard

`p12_diff_guard.py` requires six source-tree directories: `--p8-gmain`, `--p12-gmain`, `--p8-c0`, `--p12-c0`, `--a0`, and `--a1`; it also requires `--f9-allowlist` and `--output`. The F9 allowlist is JSON keyed by relative source path, with exact `a0_sha256` and `a1_sha256` values for each approved F9 change. It classifies changed files as loss plumbing, F9 intended diff, build metadata, or unexpected. Historical renderer-source changes and A0/A1 changes outside the exact F9 hashes fail. No changed CUDA file is accepted merely because its path includes F9.

Source snapshots must be complete and independently captured. A path omitted from both snapshots cannot be compared. An empty F9 allowlist is valid only if A0 and A1 source trees are identical and F9 is controlled solely by runtime switches in the same frozen binary.

## Verification

`python -m pytest tests/test_p12_guards.py -q`: **10 passed**. `python -m compileall -q` succeeded for all four tools. Tests cover GM, wall reduction and the 1.0979 regression, t-interval, contaminated attempt exclusion, missing scene, wrong reported reduction, switch failure with failure JSON, same-GPU pairing, and historical/A0-A1 unexpected diffs.

No 30K run was launched. Actual P12 source snapshots, run attestations, GPU UUIDs, and raw result JSON were not available locally, so no real campaign was certified by these tests.

# DSH TASK — WV-HIER-R1-MX Stage B-min v2 (scientific correction)

**Status:** SPECIFIED / NOT EXECUTED ON MX. This bundle contains research instructions, a PyTorch mask oracle, and local CPU unit tests. It contains no MX evidence, GPU timings, or CUDA kernel.

## Objective
Validate the HWCS gradient NMSE identity at fixed `p=0.25` on the exact same cameras/seed set, and audit structural *counts* for Native-G4 N1, mask-then-bin N2, and HWCS without presenting projected geometric counts as GPU execution time.

## Execution gate
1. Execute on one idle MX A100-PCIE-40GB **not reserved for TEMPORAL_COVERAGE-R1**. Abort if no clearly available device. Confirm device ownership and snapshot before launch. Do not assume that a single idle nvidia-smi poll authorizes preempting another job.
2. Use the frozen WV0 environment, checkpoint and `_wv0_grad.py`; no environment reinstall, compilation, changes to the historical audit, or training. Use a new output directory.
3. Record full input, source, `.so`, and output hashes. Keep raw per-camera/per-seed/per-method scalars and reconstructable masks. Absent MX access => mark BLOCKED, do not substitute a different GPU or fabricated data.

## Stop before experiment until preflight passes
- Review `STAGE_B_REVISED_SPEC.md` and run `python test_hwcs_mask_oracle.py` under MX's PyTorch.
- Patch the *new copy* of `_wv0_grad.py` to call `hwcs_mask_oracle.generate_common_uniforms` and `make_mask` and to keep one common `(U_parent, U_child)` per camera/seed across G16, G4 and both HWCS methods.
- Assert same image/target, dense gradient, SSE reduction/denominator, PSNR-independent sampling, and geometry for every method.
- `G16 == HWCS(q=p,r=1)` and `G4 == HWCS(q=1,r=p)` bitwise under common random numbers.

## Execution matrix
- Checkpoint `bicycle__iter15000`, cameras IDs `0,8`, fixed p=0.25.
- 32 predetermined paired seeds, recorded explicitly.
- 4 methods: G16, independent Native-G4 (G4), HWCS-1 q=.5 r=.5, HWCS-2 q=.75 r=1/3.
- 256 sparse full-render + HT-weighted masked SSE gradient evaluations, two dense SSE gradient references; 0 training steps; no CUDA build.
- If 32 pairs are inconclusive, report INCONCLUSIVE and ask permission for more seeds; no arbitrary DROP.

## Deliverables
- Raw row-wise CSV/Parquet with camera, seed, method, NMSE, cosine, rel-L2, dense-grad-norm, estimate-grad-norm, dot-product, input/source/mask hashes, and generated mask statistics.
- `variance_identity_check.csv` per-camera empirical `NMSE_HWCS - [(1-alpha) NMSE_G4 + alpha NMSE_G16]`, with paired-seed bootstrap CI, separate for both HWCS configs; alpha=1/3 and 1/9.
- `counts_audit.csv` with Gaussian-parent candidate counts and Gaussian-child candidate counts for the N1/N2/HWCS methods using a documented rasterization/bbox convention. Label derived geometrical counts as COUNTS/PROXY, **not GPU kernel work**. If CUDA emitted-intersection buffers are not accessible without modification, mark exact `emitted_intersections` as NOT_MEASURED.
- `BASELINE_N1_N2.md`: spell out dense-bin-then-mask (N1) vs mask-then-bin (N2), including how sort order and work differ. Do not claim HWCS dominance without N2.
- `STAGE_B_VERDICT.json` marking independent evidence statuses and any blocking limits.
- `MANIFEST_SHA256.json` and nvidia-smi snapshots before/after. Historical WV0 files unchanged.

## Decisions
1. If implementation has failed the endpoint/unbiasedness tests: DEBUG implementation; never falsify a proven mathematical identity from an implementation bug.
2. If observed HWCS NMSE significantly deviates from the same-camera convex blend despite validation: investigate loss non-separability, gradient normalization, or method mismatch; label MODEL/PROTOCOL MISMATCH, not automatic DROP.
3. If blend holds and HWCS-1 is measurably better than G16: gradient oracle PASS; claim NO GPU TIME advantage yet.
4. Do not identify cost-efficiency parameter kappa from sampling tile counts alone. `kappa` models marginal operation cost, not activation probability; its measured value requires instrumentation or a minimal CUDA timing comparison with identical environment/baselines.
5. Only propose (not run) Stage C minimal CUDA timing after oracle PASS and documented N1/N2 baselines; request separate research-lead authorization. No full training authorization from Stage B-min.

## Copy to MX
Copy this directory into a NEW run directory on MX, then follow `STAGE_B_REVISED_SPEC.md`. Use the original WV0 env and evidence as read-only references. Do not rely on this host's CPU to simulate actual MX GPU work.

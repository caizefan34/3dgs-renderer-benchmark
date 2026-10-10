# HiGS R16: Opacity semantics controlled replay

## Why

R15 sampled 1,000 projected Gaussians and reported 13.6x count disparity; first mismatch opacity=-4.595.
In the supplied source, R8 passes raw checkpoint opacity to `rasterize_gaussian_higs_frozen`, and the native forward broadcasts it to AccuTile unmodified. The CUDA kernel computes `fminf(EXTEND², 2*logf(opacity/ALPHA_THRESHOLD))`; negative inputs are invalid as alpha. `fminf` with a NaN input can choose the finite `EXTEND²` bound, potentially expanding coverage unexpectedly. Note that source from the earlier archive may not prove that R3's own benchmark used the same parameter activation or cameras.

## MX execution

Place `HIGS_R16_OPACITY_SEMANTICS_MX` on MX. Run:

```bash
cd HIGS_R16_OPACITY_SEMANTICS_MX
CUDA_VISIBLE_DEVICES=0 bash run_on_mx.sh
```

This reads the **existing R15** captured tensor file; do not re-run full 4.41M. Set `R15_CAPTURE=/actual/path/captured_projected_inputs.pt` if different. The original source and .so are never modified.

The three controlled runs have identical projected `means2d`, `radii`, `conics`, `depths`, tile grid, and tile masks, except:
- `raw`: replay original provided opacity with no changes. Must match captured raw GPU counts, else protocol mismatch.
- `sigmoid`: replace only opacity with sigmoid(opacity) to test the checkpoint-logit hypothesis.
- `sigmoid_threshold_skip`: use activated opacity and set radii zero for alpha <= threshold **only in the diagnostic replay**, to test whether invalid/noncontributing low alpha generates inflated intersections. This is NOT a production patch and is not a correctness benchmark.

Each branch runs the existing CUDA op (`sort=False`) then compares to an independent continuous ellipse-vs-tile geometric oracle. Expect bounded disagreements from AccuTile conservative discretization; distinguish large-scale changes from edge effects.

Outputs:
- `R16_OPACITY_SEMANTICS_REPORT.json` and `R16_REPLAY_COUNTS.csv`
- `oracle_raw/oracle_summary.json`
- `oracle_sigmoid/oracle_summary.json`
- `oracle_sigmoid_threshold_skip/oracle_summary.json`
- `run.log`

## Interpretation and cautions

- First validate that `raw` replay exactly matches the original R15 captured GPU counts.
- Confirm checkpoint `opacity` parameterization at the actual R3 entrypoint. Sigmoid is correct only if these saved values are logits.
- A large drop after `sigmoid` or `sigmoid_threshold_skip` supports the opacity-input/threshold mechanism; it does not prove R3 OOM is resolved.
- If count remains abnormal, audit projected scales (raw logscale vs exp), rotation normalization, camera matrices and AccuTile conic arithmetic.
- A Gaussian with activated alpha <= `ALPHA_THRESHOLD` has no pixel exceeding threshold; a legitimate full renderer fix would require matching count, emit, rasterizer and gradient semantics.
- We do **not** claim this 1k diagnostic sample corresponds to R3 4.41M four-camera workload.

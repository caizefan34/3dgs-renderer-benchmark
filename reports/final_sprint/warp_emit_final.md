# Warp-Cooperative Intersection Emit — final

## Result

`WARP_ALL` is a successful isolated pre-raster optimization on the available
three mature Reference V1 checkpoints. It uses no atomics, no compaction, and
no change to the intersection set. Pass 2 maps one warp to one Gaussian and
preserves baseline output index `k`/row-major ordering. Count pass remains
baseline; sort, offsets, rasterization, backward code, loss, and optimizer are
unchanged.

- Correctness: pass. Six real camera/checkpoint cases have exact SHA-256
  identity for all intersection structures, including pre-sort arrays. RGB and
  alpha are bitwise identical.
- 3-scene forward speedups: room **3.04×**, bicycle **2.38×**, garden **1.48×**;
  geomean **2.20×**.
- Emit speedups: **30.81×**, **20.88×**, **7.85×**; corrected geomean **17.16×**.
- Hybrid was not used: WARP_ALL is positive even for garden, and is the only
  permitted first implementation needed.
- Repeated baseline/candidate backward control passes. Cross-binary gradient
  variation matches an unmodified JIT-baseline versus prebuilt-baseline control.

## Mechanism and limitation

The mechanism is removing the extreme one-thread serial tile loop while
retaining each Gaussian's existing prefix-sum output range. Heavy-tailed
intersections/Gaussian make room and bicycle particularly favorable. Sort and
raster do not materially improve because their inputs are exactly the same.

The primary limiting factor is the residual fixed pipeline: global sort,
rasterization, projection/SH, and offset work remain after emit is reduced.
The 13-scene and training gates are unavailable because the necessary ten
checkpoints/camera datasets are absent from the audited environment.

All implementation patches, commands/logs, build logs, timing samples, raw
JSON/CSV, environment fingerprint, and discarded-run explanation are retained
under `experiments/final_sprint/warp_emit/`.

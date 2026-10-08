# A0-v2 — AdjointTensor-GS Structural Opportunity Test

Frozen date: 2026-09-29

Goal:
Test whether the exact native 3DGS alpha-compositing backward sufficient state
inside high-work raster buckets has a low-dimensional structure strong enough
to justify a future basis-moment VJP.

This is NOT:
- image-gradient downsampling;
- low-resolution training;
- skipped accepted hits;
- a new CUDA backward;
- a speed claim.

The training forward image, original loss, and native backward algebra remain exact
during data collection.

A0-v2 separates two questions:

1. ORACLE STRUCTURE:
   Is the exact per-bucket backward coefficient tensor intrinsically low rank?

2. DEPLOYABLE STRUCTURE:
   Can one fixed, scene-independent 16x16 spatial basis capture that structure
   without computing a per-bucket SVD at runtime?

Only if both are positive does A1 become worthwhile.

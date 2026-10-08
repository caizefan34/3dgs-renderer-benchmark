# HomotopyGS — H0 Start Here

## Core idea

Test whether 3DGS spends unnecessary optimization iterations fighting a hard/non-smooth renderer landscape.

HomotopyGS keeps:
- the same fine 3D Gaussian representation;
- the same full-resolution training images;
- the same original `0.8*L1 + 0.2*(1-SSIM)` objective;
- the same native renderer as the endpoint.

It changes only a continuation parameter during a short optimization trajectory and anneals it exactly back to the native renderer.

H0 is a LOCAL OPPORTUNITY TEST. It is not a full training claim.

## Frozen candidate families

1. `E2D`
   - temporarily increase the native screen-space Gaussian filter variance (`eps2d`);
   - anneal the multiplier back to 1.0.

2. `OPA`
   - temporarily reduce effective opacity with a monotone transmission temperature;
   - anneal gamma back to 1.0.

No joint E2D+OPA arm in H0.
No image downsampling.
No low-resolution loss.
No loss-weight curriculum.
No densification changes.

Read:
1. `H0_PREREG.md`
2. `H0_SCHEMA.json`
3. execute
4. finalize `H0_SUMMARY.json`
5. run `python h0_analyze.py H0_SUMMARY.json`
6. analyzer owns verdict

No production CUDA unless H0 promotes.

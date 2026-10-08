# AtomBirth-GS — AB0 Start Here

Working name: **AtomBirth-GS**.

## Core hypothesis

Standard 3DGS densification derives new primitives from existing Gaussians by clone/split rules.
AtomBirth-GS instead asks:

> Given the current multi-view full-resolution photometric objective, where should a completely new Gaussian atom be inserted so that the original native loss decreases the most?

The new atom is NOT required to have an existing Gaussian parent.

AB0 is a cheap opportunity test only.
It does NOT implement a production birth search.

## Scientific distinction

Not:
- a new split threshold;
- a residual split of an existing parent;
- dense MVS initialization;
- external depth completion;
- image-pyramid supervision.

Candidate new atoms are generated only from:
- current native renders;
- native full-resolution image adjoints / residuals;
- current camera geometry;
- current rendered depth / hit structure.

All scientific evaluation uses the original native renderer and original full-resolution loss.

Read:
1. `AB0_PREREG.md`
2. `AB0_SCHEMA.json`
3. execute
4. finalize `AB0_SUMMARY.json`
5. run `python ab0_analyze.py AB0_SUMMARY.json`
6. analyzer owns verdict

No production CUDA.
No full training.

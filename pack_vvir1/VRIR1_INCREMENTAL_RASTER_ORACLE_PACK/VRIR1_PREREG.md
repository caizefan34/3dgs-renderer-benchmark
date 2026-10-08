# Gates

G0 Remote only:
EPIC-05 GPU environment identity required.

G1 Source lock:
Freeze raster construction and Gaussian state definition.

G2 Prior art:
Check incremental BVH/raster maintenance overlap.

G3 State drift oracle:
Measure:
- xyz delta
- radius delta
- depth delta
- projected mean delta

G4 Structure delta:
Compare:
I_t and I_(t-1)

Measure:
- unchanged intersections
- added intersections
- removed intersections

G5 Economic ceiling:
Require >=70% structure unaffected before implementation.

Failure:
DROP_INCREMENTAL_HEADROOM

No CUDA before all oracle gates pass.

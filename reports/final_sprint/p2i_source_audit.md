# P2I source audit

Status: `AUDITED; NO IMPLEMENTATION AUTHORIZED BY P0`.

Runtime: gsplat 2.4.1+cu124, A100, commit `02375033388d4348376b6b607ab85f551e498a77`.

| Stage | Read/recompute | Write |
| --- | --- | --- |
| Projection | camera transform, covariance/conic, opacity extent, image test | `radii`, `means2d`, `depths`, `conics` |
| Count pass | rereads `radii`, `means2d`; clipped tile bbox | `int32 tiles_per_gauss[g]` |
| Scan | count vector | cumulative `int64` counts, `n_isects` |
| Emit pass | rereads `radii`, `means2d`, `depths`; same bbox | IDs and flattened IDs |

`ProjectionEWA3DGSFused.cu:164-208` handles `ALPHA_THRESHOLD`, extent/radius,
clipping, and projection writes. `Intersect.cpp:56-80` counts/scans and
`:95-110` emits. `IntersectTile.cu:50-84` is count mode and `:102-113` is emit.
Both use the same floor/ceil inclusive-min/exclusive-max rectangle. No AccuTile
or SNUGBOX branch is present in this audited baseline.

No renderer CUDA or wrapper source was changed. The temporary P0 extension is a
measurement-only extraction of the count branch and matched production counts.

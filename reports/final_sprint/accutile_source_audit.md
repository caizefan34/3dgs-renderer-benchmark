# AccuTile / SNUGBOX source audit

This is an existing-method integration study, not a new localization method.
The source of truth is upstream gsplat commit
`28e794ca44a4c25ffc39175370c5ee7b38bfcc36`,
[`IntersectTile.cu`](https://github.com/nerfstudio-project/gsplat/blob/28e794ca44a4c25ffc39175370c5ee7b38bfcc36/gsplat/cuda/csrc/IntersectTile.cu).
Its comments attribute SNUGBOX + AccuTile to Speedy-Splat.

Upstream provides `accutile_ellipse_intersection` and `accutile_process_tiles`.
The branch receives `conics` and `opacities`, derives the opacity-threshold
ellipse, computes its SNUGBOX, and traverses only intersecting tiles. The old
Reference V1 `IntersectTile.cu` only receives means/radii/depths and enumerates
the clipped AABB rectangle.

A minimal backport would add packed `conics` (12 bytes/visible Gaussian) and
the same opacity tensor (4 bytes/visible Gaussian) to the intersection API.
It would preserve depth, key packing, sorting, rasterization, and backward.
No backport was authorized because A0 found pixel-support false negatives.

Important semantic finding: Reference V1 currently passes stored opacity values
directly into projection. The values are logit-like (`room` min/max/mean
`-5.55/17.48/3.05`), not silently sigmoid-activated. A safe integration must
match that frozen behavior or explicitly re-establish a different baseline.

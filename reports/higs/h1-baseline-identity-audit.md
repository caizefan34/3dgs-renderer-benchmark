# H1 Baseline Identity Audit

At commit `77ab983ffe43420b2131669cb35776b883ca4c3c`, `rasterization()` passes non-null conics and opacities to `isect_tiles()` when `with_ut=False`. `IntersectTile.cu` dispatches `accutile_process_tiles`: SnugBox plus strip-based ellipse/tile enumeration. Null inputs select the classic AABB fallback. The AccuTile section is semantically equivalent to upstream `28e794ca44a4c25ffc39175370c5ee7b38bfcc36`, with HiGS tile-mask additions.

Runtime identity is BUILD_INVALID: the copied H1 `gsplat/csrc.so` exports `projection_ewa_3dgs_fused` without the wrapper's `packed` argument. The wrapper supplied `"pinhole"` in the compiled op's `calc_compensations` position. No A/B/C numerical classification is valid.

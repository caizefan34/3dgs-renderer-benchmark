# R6-B — Correctness

`experiments/r6/r6_b/test_r6b.py` compares baseline, B0, and B1 rasterizer
outputs for color, opacity, mean2d, conic, and absgrad. It explicitly executes
the stale-row sequence: iteration *t* touches row 0; iteration *t+1* touches
only row 1; row 0's mean2d gradient must be exactly zero.

`correctness_r6b.py` replays a checkpoint to compare mean2d, absgrad
densification statistics, xyz, SH, scaling, rotation, and opacity-parameter
gradients. Each comparison records `max_abs`, `mean_abs`, `relative_l2`, and
NaN/Inf status.

No R6 checkpoint/data workload exists locally, so the requested checkpoint
replay result is **not yet measured**. The machine-readable result marks this
as `not_run`; it must not be interpreted as a correctness pass.

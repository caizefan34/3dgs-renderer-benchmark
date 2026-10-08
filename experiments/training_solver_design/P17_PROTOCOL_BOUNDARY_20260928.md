# P17 multirate oracle and the full-resolution original-loss requirement

Date: 2026-09-28. Read-only protocol check of the separate `P17_P0_Multirate_Gradient_Oracle_Bundle`; no P17 result, training or GPU timing is claimed here.

P17-P0's preregistration §5 constructs d=2/4/8 cameras by reducing H/W and intrinsics, **area-downsampling GT**, and evaluating the fused loss at each coarse resolution (§8). Its per-Gaussian footprint rule routes a Gaussian to one of those coarse-loss gradients. The d=1 branch is a full-resolution reference, not the gradient used for every routed Gaussian.

Consequently a future training method that directly uses P17's routed coarse-loss gradients would not satisfy this goal's requirement that **every optimization step use the original full-effective-resolution loss**. P17-P0 can still be a frozen-state numerical feasibility study for another project's question; passing its one-step gate would not authorize a same-protocol quality/speed claim here. A full-image-loss adjoint variant would respect the scalar loss but has its own native backward state dependency and originality/cost problems, documented in `FULL_RES_ADJOINT_MULTIRATE_SOURCE_GATE_20260928.md`.

The same-hardware P16-RB bundle is a baseline-reset preregistration, not a reported result or new optimization. The current workspace contains no P16 final report or P17 verdict file. Do not borrow either planned experiment as evidence that this goal's multi-scene, multi-seed, full-process three-metric target has been reached.

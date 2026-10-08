# PSB0 Initial Prior-Art Boundary

Refresh on execution day.

Adjacent:
- Taming 3DGS / Faster-GS per-Gaussian backward: checkpoint every fixed bucket,
  then replay within bucket.
- ACM MM 2026 workload-balanced FasterGS: associative bucket-based forward/local
  compositing for load balancing.
- hardware differentiable rasterizers: different execution substrate.

Direct collision:
DROP if a public static-3DGS training method already uses an associative
prefix/suffix scan to recover the per-splat backward compositing state replacing
the checkpoint+replay formulation.

Forward associative bucket compositing alone is NOT the same claim, but it is a
serious adjacent-work constraint and must be discussed.

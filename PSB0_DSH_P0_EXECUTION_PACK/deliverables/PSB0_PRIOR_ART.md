# PSB0-P0 — Prior-Art Audit (Gate 2)

Execution-day search: 2026-10-01. Channel: arxiv API over HTTPS from EPIC-05 (Google/DDG blocked in this environment). 13 queries run; results in `PSB0_PRIOR_ART.json`.

## Direct collision (DROP condition)

Target claim: *a public static-3DGS training method that already uses an associative prefix/suffix scan to recover the per-splat backward compositing state, replacing the checkpoint+replay formulation.*

**Result: NO COLLISION FOUND.** Queries `scan+backward+compositing+splatting`, `prefix+suffix+splatting+backward`, `associative+splatting`, `bucket+backward+splatting` returned no matching method.

## Adjacent works (constraints, not collisions)

- **Taming 3DGS / FasterGS** (arXiv:2406.15643, 2024-06-21) — the checkpoint-every-fixed-bucket + within-bucket-replay backward itself. This is the *baseline* PSB0 targets, not prior art that pre-empts the scan claim.
- **LoBE-GS: Load-Balanced 3DGS** (arXiv:2510.01767, 2025-10-02) — load-balanced bucketing for *forward* compositing scheduling. Adjacent (bucket-based compositing) but forward-side, not backward state recovery.
- **ABCD: Alpha-Composited Block Coordinate Descent** (arXiv:2608.27735, 2026-08-27) — out-of-core constant-VRAM alpha-composited training. Memory-management adjacent.

## Boundary rule (per PRIOR_ART_BOUNDARY.md)

Forward associative bucket compositing alone is **not** the same claim as backward-scan state recovery, but it is a serious adjacent-work constraint. ACM MM 2026 workload-balanced FasterGS uses associative bucket compositing for *load balancing* (forward/local), which does not recover backward compositing state via prefix/suffix scan. This distinction is preserved: PSB0's claim is specifically about the *backward* state-recovery path.

## Verdict

**Gate 2 PASS — no prior-art collision.** The associative-scan backward-state-recovery formulation is not published as a static-3DGS training method as of the search date.

# IRGS Prior-Art Guard — 2026-09-30

The current search did NOT find a direct 3DGS method that maintains a persistent differentiable alpha-compositing structure across ordinary per-scene optimization iterations and exactly repairs only the state touched by changed Gaussians.

This is NOT a novelty proof.

Important nearby work:

1. FasterGS / Workload-Balanced Gaussian Splatting (ACM MM 2026)
   - bucket-based parallel local compositing
   - uses the associative alpha-compositing rule within one render
   - key distinction to audit: IRGS would persist compositing summaries ACROSS optimization steps and repair changed regions.

2. 3DGS-LM (ICCV 2025)
   - caches intermediate gradients for fast Jacobian-vector products inside LM/PCG
   - key distinction: cache serves repeated linear-solver products at one optimization state, not persistent raster-state repair across changing Gaussian parameters.

3. TideGS (2026)
   - differential streaming of changed working-set data through SSD/CPU/GPU hierarchy
   - key distinction: incremental PARAMETER TRANSFER / working-set management, not incremental differentiable rasterization.

4. This project's earlier 'Incremental Hierarchy' roadmap
   - local insert/delete updates to macro-tile offsets were already contemplated.
   - IRGS must not be framed as merely incremental hierarchy maintenance.
   - the proposed new core is persistent pixel/segment alpha-compositing summaries plus exact local forward/backward repair under sparse block-coordinate parameter updates.

Do not claim:
- first incremental rendering;
- first associative alpha compositing;
- first cache in differentiable 3DGS;
- first sparse Gaussian optimizer.

Candidate narrow claim, only after broader audit and successful experiments:
- persistent cross-iteration differentiable compositing state whose update cost is tied to changed Gaussian support/work rather than a full visible-scene rerasterization.

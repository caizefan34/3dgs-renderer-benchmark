# DSH TASK — VRIR0 Structural Coherence Oracle

## Mission
Determine whether exact/fail-closed cross-visit structural reuse is feasible in the current 3DGS training baseline. Do not optimize CUDA. Do not claim speedup from the instrumented oracle.

## Absolute prohibitions
- Do not modify `.cu`, `.cuh`, C++ extension or installed renderer kernels.
- Do not alter AABS0 or FGS-H0 branches/results.
- Do not change training hyperparameters, camera order, loss, optimizer, densification policy or renderer semantics to make VRIR0 look better.
- Do not use approximate equality to declare structural equivalence.
- Do not suppress rejects, NaNs, topology changes or memory failures.

## Phase P0 — freeze and provenance
Create a dedicated branch/worktree named `vrir0-oracle` from the canonical baseline commit.
Record:
- git commit + dirty state;
- `baseline/reference_v1/{trainer.py,config.py,gaussian_model.py}` hashes;
- installed gsplat version and exact source hash/path;
- GPU/driver/CUDA/PyTorch;
- scene source hashes;
- frozen camera sequence hashes.

Emit `outputs/P0_PROVENANCE.json`.

## Phase P1 — source audit before experiments
Trace the exact tested gsplat path from `rasterization` through tile intersection, key construction, sorting, offset encoding and rasterizer consumption.
Prove or falsify each assumption in `CERTIFICATION.md`.
Special checks:
1. What exactly does `accutile=False` do in the installed fork?
2. Is membership exactly AABB from means2d/radii?
3. What are depth key bits and sort bit range?
4. Is sort stable for equal keys? What is source emission order?
5. In dense single-camera mode, does `flatten_ids` equal Gaussian dense index?
6. Does rasterization consume only offsets + flatten IDs for discrete traversal?

If any required assumption is false, stop and emit `BLOCKED_SEMANTICS`; do not silently adapt the theorem. A revised certificate needs a new design version.

Emit `outputs/P1_SOURCE_AUDIT.md` and machine-readable `P1_SOURCE_AUDIT.json`.

## Phase P2 — oracle hook only
Add instrumentation to canonical Python training. The baseline must still execute the normal fresh intersection/sort/offset path every time.

Use a deterministic tracked-view set. Initial smoke: 2 views. Main run: 8 views/scene selected deterministically across camera indices (quantiles, including endpoints only if valid), saved to config.

On each tracked-camera visit:
- capture current fresh meta;
- compute current membership descriptor from the verified semantics;
- find previous cache entry for same camera;
- evaluate CONFIG/TOPOLOGY/NUMERIC/MEMBERSHIP certificates;
- evaluate BOTH order checks: exact O(M) adjacency scan and O(N) cached key-slack certificate;
- independently compare cached structure with the current *fresh* `flatten_ids` and `isect_offsets`;
- append one JSONL record;
- replace the cached entry with current fresh structure for the next revisit.

Store large structure tensors on CPU. Enforce a byte cap. If cap is exceeded, record `CACHE_BUDGET_REJECT` rather than dropping data silently.

## Phase P3 — smoke falsification
Scene: room.
Window: start from a suitable post-15k checkpoint if available; otherwise run canonical continuation without altering state.
Tracked views: 2.
Require at least 20 same-view revisit comparisons.

Stop criteria:
- any false accept;
- any unknown source/tie semantics;
- any mismatch between computed membership descriptor and fresh baseline structure attributable to the oracle model.

If smoke passes, continue.

## Phase P4 — main structural coherence collection
Scenes: room, bicycle, garden.
Primary iterations: 15001–30000.
Tracked views: 8 deterministic cameras/scene.
Optional diagnostic collection: 0–15000, clearly marked diagnostic.

Do not time the instrumented oracle as an implementation benchmark.

Produce per-observation `outputs/<scene>/vrir0_records.jsonl`.

## Phase P5 — analysis
Run `scripts/vrir0_analyze.py` or an equivalent audited analyzer.
Report by scene and 5k window:
- revisit gap distribution;
- topology eligibility;
- truth_same rate;
- C_MEM, C_ORD_EXACT_SCAN, C_ORD_SLACK_N, and full fast-certificate hit rates;
- false accepts and false rejects;
- tile-local unchanged fraction;
- adjacent order violation rate;
- n_isects/cache bytes;
- drift quantiles;
- baseline T_struct and T_iter provenance;
- optimistic Amdahl upper bound.

Do not turn oracle Python overhead into a projected implementation cost. Record it separately only as instrumentation overhead.

## Phase P6 — verdict
Apply `FAIL_CLOSED_GATES.md` mechanically.
Output exactly one of:
- KEEP_FOR_MINIMAL_IMPLEMENTATION
- REDESIGN_TO_TILE_GRANULAR_VRIR1
- DROP
- BLOCKED

Generate:
- `VRIR0_VERDICT.json`
- `VRIR0_REPORT.md`
- `VRIR0_PRIOR_ART_REFRESH.md`
- `MANIFEST_VRIR0.json` with SHA256 of every evidence file.

## Required report discipline
Use OBSERVED / EVIDENCE / HYPOTHESIS / TEST / CONCLUSION sections.
Separate:
- measured fact;
- derived quantity;
- assumption;
- speculative implementation projection.

A zero false-accept result is necessary but not sufficient for KEEP; opportunity and memory gates can still DROP the candidate.

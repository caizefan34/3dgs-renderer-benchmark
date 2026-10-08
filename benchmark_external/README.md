# External Training Comparison Protocol (v0)

Date: 2026-10-01
Status: protocol DEFINITION (package prepared; external baseline runs not yet executed).

## Purpose
This package defines a TRAINING-time comparison protocol for 3DGS systems. It is
DISTINCT from `benchmark/` (the rendering/inference suite + `native_training`
track). This protocol measures training cost, quality, and exactness, and never
shares a ranking with the rendering track.

It was prepared as part of the DSH Parallel Tasks Packet 20261001, task 3
(Benchmark Plan).

## Files
- `protocol.json` — common recipe, measurement boundaries, evidence tiers, ranking rules, promotion gates.
- `baselines.json` — the baseline roster (external + internal + conditional LAVOS).
- `correctness.json` — state hash / topology hash / lineage consistency definitions (bitwise, no tolerance).
- `result.schema.json` — JSON Schema for one comparison result record.
- `README.md` — this file.

## Baselines (see baselines.json)
- External reference: INRIA 3DGS (graphdeco).
- External public: FastGS, Sparse Adam, gsplat.
- Internal exact: VOS-GS, ReproTopo0.
- Internal exact (conditional): LAVOS0 — included ONLY if promoted.
- Closed/positioning-only: VTS0 (CLOSED), DetGS0 (positioning only).

## Metrics (see protocol.json + result.schema.json)
- Efficiency: training_time_s, iteration_time_ms (mean/p50/p90), optimizer_time_ms, peak_vram_mb, gaussian_count_final.
- Quality: PSNR, SSIM, LPIPS (+ holdout variants on the i%8==0 split).
- Correctness: state_hash, topology_hash, lineage_consistency vs the frozen C0 dense reference.

## Rules
- Never mix evidence tiers (measured / reproduced / paper_reported).
- Exactness gates are SEPARATE from speed: a baseline that fails correctness is EXACTNESS_FAIL and is not ranked against exact methods on speed.
- No threshold relaxation; gate changes require a new protocol version.
- LAVOS is a baseline only if promoted; otherwise NOT_PROMOTED.
- VTS0 is CLOSED and is not a baseline.

## Relation to the other packet tasks
- VTS0 (task 1) measured extra_safe_skip_headroom = 0.0 < 10% => CLOSED. Recorded in protocol promotion_gates.
- DetGS0 (task 2) is positioning-only (NO_PUBLIC_COLLISION vs public; unified idea occupied by LAVOS0 internally). Recorded in baselines.json closed_or_positioning_only.

## Execution status
The protocol, baselines, correctness definitions, and result schema are defined.
Running the external baselines (FastGS / Sparse Adam / gsplat) on EPIC-05 to
produce measured rows is the next step; it requires their official repos cloned
and built per `protocol.json` recipe. This package is the comparison protocol
itself, ready to drive those runs.

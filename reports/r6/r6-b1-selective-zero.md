# R6-B1 — Previous-Touched Selective Clear

## Invariant and implementation

The next backward clears `prev_touched_ids`, never `current_touched_ids`.
After each raster backward, the immutable `flatten_ids` forward metadata is
retained as the next previous touched set. The B1 clear kernel writes zero to
all 11 rasterizer-gradient floats for every previous ID before current
accumulation begins. Duplicate IDs are safe because every store is zero.

`flatten_ids` is a conservative touched superset: an intersection rejected by
an alpha or mask branch may still be cleared, which preserves exactness.

`v_means2d_abs` is not touched until the following backward begins; therefore
the existing densification-stat consumer can read it after backward normally.

## Topology handling

- Growth or any logical shape change: full clear.
- Explicit `r6b_invalidate()` after clone, split, or prune: full clear.
- `runtime.install_topology_guard()` and `training_sanity_r6b.py` provide the
  required hook without changing optimizer or densification semantics.
- Packed rendering falls back to the original allocation path because it does
  not have the stable unpacked `[C,N]` row identity used here.

## Status

The implementation is present but the real-workload gather/scatter cost has
not been measured locally; no claim is made that B1 wins B0.

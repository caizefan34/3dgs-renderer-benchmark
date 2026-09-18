# C17-1 CUDA design

`C17_BASELINE` remains the default. `C17_TILE_LOCAL` selects a new
`intersect_tile_c17` binding.

1. A Gaussian-parallel count kernel enumerates its tile rectangle, writes the
   unchanged `tiles_per_gauss`, and atomically increments `tile_counts`.
2. An exclusive scan makes exact tile offsets and the sum sizes the arena.
3. A second Gaussian-parallel kernel atomically claims one offset-local slot,
   directly yielding tile-contiguous `flatten_ids`.
4. `DeviceSegmentedRadixSort` sorts only each existing tile segment. Its
   temporary key is `(float32-depth-bit-pattern, input-index)`, never the old
   `(tile, depth)` key. The index explicitly preserves stable baseline ordering
   on exact depth ties.

This implementation deliberately uses a single robust segmented fallback while
the required histogram determines whether a later tiered local sorter is worth
adding. No capacity allocation is used.

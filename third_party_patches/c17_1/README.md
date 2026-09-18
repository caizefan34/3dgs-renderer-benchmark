# C17-1 patch payload

`apply_c17_1_patch.py` patches a disposable gsplat 1.5.3 source tree. It adds
an opt-in `intersect_tile_c17` binding only; the upstream `intersect_tile`
binding and default Python path are unchanged. `C17_TILE_LOCAL` selects it.

The implementation intentionally uses exact allocation: tile counts, an
exclusive scan, and a contiguous arena sized to the observed sum. The sorting
primitive is segmented only after direct construction; no `(tile_id, depth)`
key is ever built on the C17 path.

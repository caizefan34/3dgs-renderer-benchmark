# AccuTile A0 correctness gate

Status: `FAIL — STOP BEFORE BACKPORT`.

The AccuTile pair set is a subset of the AABB pair set for every checked pair:
subset failures were zero in room, bicycle, and garden. This alone is not enough
for exact localization.

For every removed pair, A0 first tested continuous ellipse support and then
tested the actual Reference V1 raster pixel centers (`x+0.5`, `y+0.5`) with its
same alpha threshold. A0.5 then re-ran the 26 surviving cases in CUDA float32
with the raster's `__expf`, alpha clamp, and `<` cutoff. Removed tiles containing
at least one threshold-passing pixel were found:

| Scene | Removed pairs | Continuous candidates | Pixel-support false negatives |
| --- | ---: | ---: | ---: |
| room | 13,140,607 | 6 | 6 |
| bicycle | 9,379,004 | 12 | 9 |
| garden | 5,007,840 | 12 | 11 |

Examples and the minimum continuous `q - t` margin are in the raw JSON. Since
even one false-negative tile violates the requested conservative contract, no
renderer backport, render parity, gradient parity, or timing benchmark is valid.

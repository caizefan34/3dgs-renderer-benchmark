# AccuTile baseline audit

`ACCUTILE_STATUS = ABSENT`

The actual training entry point is `baseline/reference_v1/trainer.py`, which imports
`gsplat.rasterization`.  At runtime this resolves to `C:\Users\36570\miniconda3\Lib\site-packages\gsplat\rendering.py`, not a repository vendored copy.

| Item | Observed value |
| --- | --- |
| repository commit / branch | `fd6075d61deba224cdb9456ce1412898636d27b2` / `master` before this task; work branch `codex/accutile-backport` |
| actual gsplat | PyPI `1.4.0`, source tag `v1.4.0` = `4d3a3b69db4de0326f983ccf7b7b255271a17b01` |
| CUDA / PyTorch | CUDA runtime 13.0 / PyTorch 2.13.0+cu130; nvcc 13.3 |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU, SM 12.0 |
| actual source path | `gsplat/rendering.py` -> `gsplat/cuda/_wrapper.py:isect_tiles` -> `gsplat/cuda/csrc/isect_tiles.cu` |

The frozen kernel first computes a 3.33-sigma screen-space radius in projection,
then enumerates every tile in the clipped square AABB.  It emits depth/tile keys,
radix-sorts them, encodes offsets, and passes the sorted flat IDs to forward and
backward rasterizers.  It has no conic/opacity inputs and no ellipse-vs-tile test.

Upstream `28e794ca44a4c25ffc39175370c5ee7b38bfcc36` contains optional conic and
opacity inputs plus AccuTile/SNUGBOX.  The minimal backport is
`third_party_patches/gsplat-1.4.0-accutile.patch`; it leaves the no-input AABB
path unchanged and enables the conservative path only when `accutile=True`.

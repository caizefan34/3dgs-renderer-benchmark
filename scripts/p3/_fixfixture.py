#!/usr/bin/env python3
"""Fix the P3-H smoke fixture construction.

`_native_forward_capture` calls the `gsplat.cuda._wrapper.isect_tiles` wrapper,
which invokes `torch.ops.gsplat.intersect_tile` with a 13-arg signature that
OMITS the optional `tile_mask` argument. The frozen env's installed/compiled
gsplat core registers `intersect_tile` with a signature that REQUIRES
tile_mask, so the call raises:

  gsplat::intersect_tile() is missing value for argument 'tile_mask'

This is the exact ABI that p3_0_har_measure.py handles by calling the op
directly with a try/except over both the with-tile_mask (trailing None) and
without-tile_mask forms. Replicate that proven two-signature call at the single
capture site. All downstream captured-tensor shapes are unchanged (only the
intersection computation path is swapped from wrapper to direct op), so the
already-built variant .so consume the identical fixture. No rebuild needed.
"""
import sys

F = ("/mnt/storage_pool/liaoyuanjun/higs_p3h_worktree_gsplat"
     "/experimental/render/functional/gaussian_inference.py")
src = open(F, encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    c = src.count(old)
    if c != n:
        raise SystemExit("pattern count %d != %d for:\n%r" % (c, n, old))
    src = src.replace(old, new)


OLD = (
    "            tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(\n"
    "                means2d,\n"
    "                radii,\n"
    "                depths,\n"
    "                tile_size,\n"
    "                tile_width,\n"
    "                tile_height,\n"
    "                packed=False,\n"
    "                n_images=C,\n"
    "                image_ids=None,\n"
    "                gaussian_ids=None,\n"
    "                conics=conics,\n"
    "                opacities=opacities_bc,\n"
    "            )\n"
    "            ctx.n_isects_sampled = int(isect_ids.numel())\n"
)

NEW = (
    "            # (P3-H diag fix) Call the native intersect op DIRECTLY so the\n"
    "            # signature matches whatever gsplat.core is loaded at runtime.\n"
    "            # The frozen core requires a trailing tile_mask arg; older cores\n"
    "            # omit it. Mirror p3_0_har_measure's proven try/except over both.\n"
    "            try:\n"
    "                _, isect_ids, flatten_ids = torch.ops.gsplat.intersect_tile(\n"
    "                    means2d.contiguous(), radii.contiguous(), depths.contiguous(),\n"
    "                    conics.contiguous(), opacities_bc.contiguous(), None, None, C,\n"
    "                    tile_size, tile_width, tile_height, True, False, None)\n"
    "            except RuntimeError:\n"
    "                _, isect_ids, flatten_ids = torch.ops.gsplat.intersect_tile(\n"
    "                    means2d.contiguous(), radii.contiguous(), depths.contiguous(),\n"
    "                    conics.contiguous(), opacities_bc.contiguous(), None, None, C,\n"
    "                    tile_size, tile_width, tile_height, True, False)\n"
    "            ctx.n_isects_sampled = int(isect_ids.numel())\n"
)

rep(OLD, NEW, 1)

open(F, "w", encoding="utf-8").write(src)
print("FIXFIXTURE OK")
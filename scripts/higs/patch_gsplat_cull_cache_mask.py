import sys

from pathlib import Path

# Round-60 incremental patch on top of the round-59 cull-cache-key patch
# (patch_gsplat_cull_cache_key.py).  The round-59 cache slot is a 2-tuple
# (vis_ids, fwd_count); storing the already-computed union-visibility bool
# mask [N] as the middle element gives the benchmark harness a zero-cost
# per-Gaussian "did the latest train forward render this Gaussian" signal
# for the cull-masked Adam step (round-60 lever).  Slot layout becomes
# (vis_ids, visible_mask, fwd_count); cache-hit code keeps using slot[0].
#
# Usage: python patch_gsplat_cull_cache_mask.py [target_gaussian_inference.py]
# Default target is the remote runtime copy under /root/3dgs-roadmap-matrix.

DEFAULT = Path("/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat/gsplat/experimental/render/functional/gaussian_inference.py")
path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
src = path.read_text(encoding="utf-8")


def once(old, new, tag):
    global src
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"[{tag}] expected 1 occurrence, got {n}")
    src = src.replace(old, new)
    print(f"OK {tag}")


# P13: keep the union-visibility mask in the train/eval cache slot.  The slot
# layout moves from (vis_ids, fwd_count) to (vis_ids, visible_mask, fwd_count),
# so the cache-hit freshness check must read slot[2] instead of slot[1].
once(
    "    if slot is not None and (handle._fwd_count - slot[1]) < interval:\n",
    "    if slot is not None and (handle._fwd_count - slot[2]) < interval:\n",
    "P13_cache_hit_fwdcount",
)
once(
    "    handle._cull_cache[cache_key] = (vis_ids, handle._fwd_count)\n",
    "    handle._cull_cache[cache_key] = (vis_ids, _vis_mask, handle._fwd_count)\n",
    "P13_cache_slot_mask",
)
path.write_text(src, encoding="utf-8")
print("PATCHED_ALL_OK")
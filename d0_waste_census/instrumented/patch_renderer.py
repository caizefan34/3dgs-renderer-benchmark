#!/usr/bin/env python3
"""Apply the D0 metadata-only patches to a COPY of the frozen C0 worktree.

Usage:  python3 patch_renderer.py <worktree_root> [diff_out_path]

Patches gsplat/experimental/render/functional/gaussian_inference.py with five
insertions that only EXPOSE already-computed tensors (tiles_per_gauss, per-
visible depths, visible-row pixel means2d) through the existing metadata dicts
and `_HigsAutogradFunction.last_forward_metadata`.  No tensor is modified, no
op is reordered, no RNG is consumed.
"""
import difflib
import os
import sys

REL = "gsplat/experimental/render/functional/gaussian_inference.py"

PATCHES = [
    # R1: expose per-visible tile counts + depths at the intersect site
    (
        """            ctx.n_isects_sampled = int(isect_ids.numel())
""",
        """            ctx.n_isects_sampled = int(isect_ids.numel())
            # D0 instrumentation (metadata only): expose already-computed
            # per-visible tensors; no tensor value or op order changes.
            ctx.d0_tiles_per_gauss = tiles_per_gauss.detach()
            ctx.d0_depths_vis = depths.reshape(-1).detach()
""",
    ),
    # R2: expose visible-row pixel-space means2d (captured[0])
    (
        """        )
        return render_colors, render_alphas, captured
""",
        """        )
        ctx.d0_means2d_px = captured[0].detach()
        return render_colors, render_alphas, captured
""",
    ),
    # R3: metadata dict of the render path
    (
        """            "densification_radii": densification_radii,
            "visible_gaussian_ids": master_visible_ids,
        }
        return frame[None], alpha_ch[None]
""",
        """            "densification_radii": densification_radii,
            "visible_gaussian_ids": master_visible_ids,
            "d0_tiles_per_gauss": getattr(ctx, "d0_tiles_per_gauss", None),
            "d0_depths_vis": getattr(ctx, "d0_depths_vis", None),
            "d0_means2d_px": getattr(ctx, "d0_means2d_px", None),
            "d0_f9_enabled": bool(getattr(ctx, "f9_enabled", False)),
        }
        return frame[None], alpha_ch[None]
""",
    ),
    # R4: metadata dict of the no-visible early return
    (
        """                "visible_gaussian_ids": torch.empty(
                    0, dtype=torch.long, device=means.device
                ),
            }
            if C == 1:
""",
        """                "visible_gaussian_ids": torch.empty(
                    0, dtype=torch.long, device=means.device
                ),
                "d0_tiles_per_gauss": None,
                "d0_depths_vis": None,
                "d0_means2d_px": None,
                "d0_f9_enabled": False,
            }
            if C == 1:
""",
    ),
    # R5: dynamic-entry return metadata (_higs_dynamic_forward; the entry the
    # C0 recipe calls).  The leading `"scene_version": scene_version,` line
    # makes the anchor unique -- _higs_frozen_forward's metadata dict is
    # otherwise identical here but is never on the dynamic path.
    (
        """            "n_gaussians": N_total,
            "scene_version": scene_version,
            "backward_backend": fwd_meta["backward_backend"],
            "native_available": fwd_meta["native_available"],
            "n_visible": fwd_meta["n_visible"],
            "culling_ratio": fwd_meta["culling_ratio"],
            "topology_rebuilt": fwd_meta["topology_rebuilt"],
            "packed_dtype": fwd_meta["packed_dtype"],
            "sh_compression_mode": fwd_meta.get("sh_compression_mode"),
            "render_mode": render_mode,
            "use_higs_culling": use_higs_culling,
            "higs_preview": higs_preview,
            "sampled_tile_ratio": fwd_meta.get("sampled_tile_ratio", 1.0),
            "tile_mask": fwd_meta.get("tile_mask"),
            "n_isects": fwd_meta.get("n_isects", 0),
            "n_isects_full": fwd_meta.get("n_isects_full", 0),
            "cull_refresh_interval": cull_refresh_interval,
""",
        """            "n_gaussians": N_total,
            "scene_version": scene_version,
            "backward_backend": fwd_meta["backward_backend"],
            "native_available": fwd_meta["native_available"],
            "n_visible": fwd_meta["n_visible"],
            "culling_ratio": fwd_meta["culling_ratio"],
            "topology_rebuilt": fwd_meta["topology_rebuilt"],
            "packed_dtype": fwd_meta["packed_dtype"],
            "sh_compression_mode": fwd_meta.get("sh_compression_mode"),
            "render_mode": render_mode,
            "use_higs_culling": use_higs_culling,
            "higs_preview": higs_preview,
            "sampled_tile_ratio": fwd_meta.get("sampled_tile_ratio", 1.0),
            "tile_mask": fwd_meta.get("tile_mask"),
            "n_isects": fwd_meta.get("n_isects", 0),
            "n_isects_full": fwd_meta.get("n_isects_full", 0),
            "d0_tiles_per_gauss": fwd_meta.get("d0_tiles_per_gauss"),
            "d0_depths_vis": fwd_meta.get("d0_depths_vis"),
            "d0_means2d_px": fwd_meta.get("d0_means2d_px"),
            "d0_f9_enabled": bool(fwd_meta.get("d0_f9_enabled", False)),
            "cull_refresh_interval": cull_refresh_interval,
""",
    ),
]


def main():
    root = sys.argv[1]
    path = os.path.join(root, REL)
    with open(path, encoding="utf-8") as f:
        orig = f.read()
    text = orig
    for i, (old, new) in enumerate(PATCHES):
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"renderer patch R{i + 1} anchor occurs {n} "
                             f"times (must be 1)")
        text = text.replace(old, new, 1)
    if text == orig:
        raise SystemExit("renderer: no change produced")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    if len(sys.argv) > 2:
        diff = "".join(difflib.unified_diff(
            orig.splitlines(keepends=True), text.splitlines(keepends=True),
            fromfile="frozen/" + REL, tofile="d0/" + REL, n=3))
        with open(sys.argv[2], "w", encoding="utf-8", newline="\n") as f:
            f.write(diff)
        print(f"renderer patched ({len(diff.splitlines())} diff lines)")
    else:
        print("renderer patched")
    import ast
    ast.parse(text)
    print("ast-parse OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

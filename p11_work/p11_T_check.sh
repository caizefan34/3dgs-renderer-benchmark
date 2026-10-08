#!/bin/bash
cd /root/epic05_p11
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
d = json.load(open('dumps/P11_ACTIVE_INTERACTION_AUDIT.json'))
for scene, cells in d["per_scene"].items():
    for label, c in cells.items():
        print(f"=== {scene}/{label} ===")
        print(f"  tile_candidate_interactions={c['tile_candidate_intersections']:,}"
              if 'tile_candidate_intersections' in c else
              f"  tile_candidate_interactions={c.get('tile_candidate_interactions', c.get('n_isects', '?')):,}")
        for k in ("exact_contributing_interactions", "pixel_traversed_by_bwd_est",
                  "warp_traversed_by_bwd_est", "frac_warp_slack",
                  "frac_noncontrib_in_range", "n_tiles_nonzero",
                  "total_tiles", "n_gaussians", "n_visible"):
            if k in c:
                v = c[k]
                print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v:,}")
        for k in ("validation_alphas_match_forward", "validation_max_alpha_diff"):
            if k in c:
                print(f"  {k}: {c[k]}")
        r = c.get("conservative_residual_removable_pct_e2e")
        if r is not None:
            print(f"  conservative_residual_pct_e2e: {r:.3f}")
print("=== T-GATE ===")
print(json.dumps(d["t_gate"], indent=1))
EOF

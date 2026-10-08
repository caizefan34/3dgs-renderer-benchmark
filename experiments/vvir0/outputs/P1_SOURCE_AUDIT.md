# VRIR0 Phase P1 — Source Audit (Gate G1)

## Installed vs assumed
- Installed gsplat: **1.4.0** (stock)
- Pack/CERTIFICATION assumed: 1.5.3_fork
- Version deviation: True

Pack/CERTIFICATION assumes a gsplat 1.5.3 fork with per-axis radii [C,N,2]. Installed is stock gsplat 1.4.0 with radii [C,N]. The AABB membership semantics are IDENTICAL (single radius = 3*sigma used for both axes); only the baseline densification visibility_filter shape differs, which is OFF in the 15001-30000 oracle window. VRIR0-AABB certification remains valid on 1.4.0.

## Special checks (CERTIFICATION.md)

| # | Check | Result | File |
|---|---|---|---|
| 1 | accutile=False selects exact AABB membership | VERIFIED | cuda/csrc/isect_tiles.cu |
| 2 | Membership == AABB from means2d/radii | VERIFIED | cuda/csrc/isect_tiles.cu |
| 3 | Depth key bits + sort bit range | VERIFIED | cuda/csrc/isect_tiles.cu |
| 4 | Sort stable for equal keys; emission order | VERIFIED_WITH_CAVEAT | cuda/csrc/isect_tiles.cu |
| 5 | flatten_ids == dense index (single-cam dense) | VERIFIED | cuda/csrc/isect_tiles.cu |
| 6 | Rasterizer consumes only offsets + flatten IDs | VERIFIED | cuda/csrc/rasterize_to_indices_in_range.cu |

## Overall: PASS

All 6 CERTIFICATION special checks verified on installed gsplat 1.4.0. accutile=False selects exact AABB membership; key=camera|tile|depth(32-bit); flatten_ids=dense index; rasterizer consumes offsets+flatten_ids for discrete traversal; sort is stable with emit_rank=index. VRIR0-AABB certification is VALID on the installed fork. No BLOCKED_SEMANTICS.

## Depth definition
depths[idx] = mean_c.z (view-space z) from fully_fused_projection_fwd.cu line 187. Positive for in-front-of-camera Gaussians; radius=3*sigma, radii<=0 => culled (no tiles).

## Additional findings
1. Baseline deviation (does NOT affect oracle window): baseline/reference_v1/gaussian_model.py add_densification_stats expects means2d.absgrad and densify_and_prune uses radii.max(dim=-1), assuming radii shape [C,N,2]. Installed gsplat 1.4.0 returns radii [C,N] (single radius). This breaks the densification path (iters<15000) but is irrelevant to VRIR0 because (a) the oracle window 15001-30000 has densification OFF, and (b) continuation starts from a post-15k trained checkpoint. Recorded as a provenance deviation.
2. Toolchain note: building the gsplat CUDA extension on this Windows/cp936 host requires vcvars64.bat + VSLANG=1033 and a runtime monkeypatch of torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS to ('cp936','replace') because torch decodes cl.exe output with the OEM codec while cl.exe emits cp936 Chinese. Build-tooling only; does not touch renderer kernels or training semantics.

## Source hashes
`json
{
  "isect_tiles.cu": "f50b9963a93b091e60f69f532448d0b185ae4f24a12867e1c33f1989392f33cf",
  "rasterize_to_indices_in_range.cu": "da0b0b9d0aeafbdd0331423238615f002f930f2c34d217fbcb2312df21aab164",
  "fully_fused_projection_fwd.cu": "8217e7897f1bc128088cb9b2b3abc496a7bc4af29d683a5d258ee4a877617607",
  "rendering.py": "bae7a73d0cd8bd2ddc44ae95bb9bcd293b4e0f5d2628b1a8f9a2e9873e88f1f3"
}
`

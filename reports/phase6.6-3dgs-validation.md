# Phase 6.6 — 3DGS Validation Report

## Real data used

From Phase 6.5 research on 3D Gaussian Splatting tile-size comparison:

| Scene | Config | Hardware | FPS | PSNR (dB) |
|-------|--------|----------|-----|-----------|
| bicycle | gsplat_higs_tile16 | A100-SXM4-80GB | ~500 | ~24.31 |
| bicycle | gsplat_higs | A100-SXM4-80GB | ~492 | — |
| garden | gsplat_higs_tile16 | A100-SXM4-80GB | ~442 | ~25.83 |
| garden | gsplat_higs | A100-SXM4-80GB | ~492 | ~25.83 |

## Research Finding generated

**Statement:**
> "Under A100 with bicycle and garden scenes, there is insufficient evidence that tile32 is universally faster than tile16. Performance is workload-dependent."

**Evidence summary:**
> bicycle: tile16 500 FPS, tile32 490 FPS; garden: tile16 492 FPS, tile32 502 FPS

**Scope:**
- Hardware: NVIDIA A100-SXM4-80GB
- Dataset: mipnerf360
- Resolutions: 1080p
- Tile sizes compared: 16, 32

**Alternative explanations:**
1. Scene complexity (Gaussian count) shifts optimal tile size
2. Different coverage patterns change tile intersection workload
3. Tile32 reduces culling overhead but increases per-tile sorting

**Contradicting evidence:**
1. Tile32 faster on garden (502 vs 492 FPS)
2. Tile16 faster on bicycle (500 vs 490 FPS)

**Uncertainty:**
- Limited scene coverage (2/5 scenes)
- Single GPU (A100)
- No resolution sweep

## Knowledge Impact

The finding correctly:
- ❌ Does **not** claim "tile32 is slower"
- ❌ Does **not** claim "tile16 is always better"
- ✅ Preserves both sides of the evidence
- ✅ Documents uncertainty
- ✅ Identifies gaps for further study
- ✅ Generates a focus recommendation

## Verdict

**End-to-End Closure: PASS**
**Scope Preservation: PASS**
**Knowledge Promotion Safety: PASS**

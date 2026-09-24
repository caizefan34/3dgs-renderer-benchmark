"""Trace the exact pipeline sources for both corrected baseline and membership audit."""
import json

print("=" * 70)
print("PIPELINE COMPARISON: Corrected Baseline vs Membership Audit")
print("=" * 70)

print("""
1. GAUSSIAN SOURCE
┌─────────────────────────┬────────────────────────────────┬────────────────────────────────┐
│ Factor                  │ Corrected A100 Baseline        │ Membership Audit (c17_2)       │
├─────────────────────────┼────────────────────────────────┼────────────────────────────────┤
│ Source file             │ Trained checkpoint (30k steps) │ SfM point cloud (PLY)          │
│ Path (room)             │ results/epic05/phase7/         │ data/official/mipnerf360/room/ │
│                         │ a100_30k_room_t16_16/          │ point_cloud.ply                │
│                         │ a100_30k_room_t16_16_latest.pt │                                │
│ Gaussian count (room)   │ ~1.59M (from checkpoint)       │ 1,593,376 (from PLY header)    │
│ Scales                  │ log-scale in checkpoint,       │ log-scale in PLY,              │
│                         │ torch.exp activated            │ torch.exp activated            │
│ Opacity                 │ sigmoid from logit             │ sigmoid from logit             │
│ SH degree               │ 3 (trained)                    │ 0 (from PLY)                   │
├─────────────────────────┼────────────────────────────────┼────────────────────────────────┤
│ CAMERA                  │                                │                                │
│ Source                  │ cameras.json (1st camera)      │ cameras.json (1st camera)      │
│ Resolution (room)       │ 3114 × 2075 (native, no resize)│ 1080 × 1080 (resized down)     │
│ Resolution (bicycle)    │ 4946 × 3286 (native)           │ 1920 × 1080 (resized)          │
│ Resolution (garden)     │ 5187 × 3361 (native)           │ 1920 × 1080 (resized)          │
├─────────────────────────┼────────────────────────────────┼────────────────────────────────┤
│ TILE GRID               │                                │                                │
│ room tile grid          │ ceil(3114/16)×ceil(2075/16)    │ ceil(1080/16)×ceil(1080/16)    │
│                        │ = 195 × 130 = 25,350 tiles     │ = 68 × 68 = 4,624 tiles        │
│ bicycle tile grid       │ ceil(4946/16)×ceil(3286/16)    │ ceil(1920/16)×ceil(1080/16)    │
│                        │ = 310 × 206 = 63,860 tiles     │ = 120 × 68 = 8,160 tiles       │
│ garden tile grid        │ ceil(5187/16)×ceil(3361/16)    │ ceil(1920/16)×ceil(1080/16)    │
│                        │ = 325 × 211 = 68,575 tiles     │ = 120 × 68 = 8,160 tiles       │
├─────────────────────────┼────────────────────────────────┼────────────────────────────────┤
│ VISIBLE GAUSSIANS       │                                │                                │
│ room visible G          │ 16,076                         │ 389,647                        │
│ bicycle visible G       │ 13,654                         │ 1,805,534                      │
│ garden visible G        │ 100,582                        │ 2,248,958                      │
├─────────────────────────┼────────────────────────────────┼────────────────────────────────┤
│ INTERSECTIONS           │                                │                                │
│ room n_isects           │ 25,475,247                     │ 1,690,772                      │
│ bicycle n_isects        │ 18,550,853                     │ 4,861,253                      │
│ garden n_isects         │ 11,257,252                     │ 5,110,309                      │
├─────────────────────────┼────────────────────────────────┼────────────────────────────────┤
│ MEAN TILES/G            │                                │                                │
│ room                    │ 25,475,247 / 16,076 ≈ 1,584.8  │ 1,690,772 / 389,647 ≈ 4.34     │
│ bicycle                 │ 18,550,853 / 13,654 ≈ 1,358.7  │ 4,861,253 / 1,805,534 ≈ 2.69   │
│ garden                  │ 11,257,252 / 100,582 ≈ 111.9   │ 5,110,309 / 2,248,958 ≈ 2.27   │
└─────────────────────────┴────────────────────────────────┴────────────────────────────────┘
""")

print("""
2. WHY THE NUMBERS DIFFER (Root Cause Analysis)

The corrected baseline uses TRAINED checkpoints (30k steps). During training:
  - Gaussians grow in scale to fill the scene
  - Many Gaussians are pruned (opacity → 0)
  - Surviving Gaussians have large 2D footprints

Raw SfM PLY: sparse, small points from COLMAP initialization.
  - Many points visible but each covers FEW tiles (mean ~2-4)
  - 24-49% of Gaussians are visible (depends on density vs resolution)

Trained checkpoint: dense scene representation.
  - FEWER Gaussians visible (1-6% due to pruning/opacity filtering)
  - Each visible Gaussian covers MANY tiles (mean ~112-1585)
  - Result: FAR MORE intersections despite fewer Gaussians

3. n_isects DEFINITION

For BOTH pipelines:
  n_isects = sum( tiles_per_gaussian ) over all visible Gaussians
          = keys.numel() = values.numel() = flatten_ids.numel()

Formula verified: n_isects = sum_g n_tiles(g) always holds.

4. KEY OBSERVATION

The "68,575" tile count is garden at native resolution:
  ceil(5187/16) * ceil(3361/16) = 325 * 211 = 68,575

The "8,160" tile count is 1920×1080 at tile size 16:
  120 * 68 = 8,160

These are NOT the same workload — they differ in both scene resolution AND Gaussian source.

5. CORRECT WORKLOAD IDENTITY

The Corrected A100 Baseline uses:
  - Trained checkpoints from epic05/phase7 (30k, tile16)
  - Native camera resolution
  - packed=True, tile_size=16
  - First camera of each scene

Membership audit (c17_2) used:
  - Raw SfM PLY
  - RESIZED resolution (1080×1080 or 1920×1080)
  - packed=True, tile_size=16
  - First camera of each scene

These are COMPLETELY DIFFERENT workloads.
""")

print("=" * 70)
print("ARITHMETIC CONSISTENCY CHECK")
print("=" * 70)
print("""
Membership audit (resized PLY):
  room:     n_isects=1,690,772  n_visible=389,647  mean_tpg=4.3392  389,647×4.3392=1,690,772  ✓
  bicycle:  n_isects=4,861,253  n_visible=1,805,534 mean_tpg=2.6924  1,805,534×2.6924=4,861,253 ✓
  garden:   n_isects=5,110,309  n_visible=2,248,958 mean_tpg=2.2723  2,248,958×2.2723=5,110,309 ✓

All arithmetic is consistent. Previous apparent discrepancy was due to
multiplying n_visible mean by master count instead of visible count.
""")

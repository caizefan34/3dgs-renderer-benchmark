# AccuTile benchmark

Only the local frozen Room checkpoint was available on this host.  The requested
bicycle and garden checkpoints were not present, so they are intentionally null in
the machine-readable result rather than inferred from older experiments.

Room workload: `results/epic05/phase7/phase7_room_30k_16/phase7_room_30k_16_latest.pt`,
camera 0, 1,000,684 Gaussians, 3114x2075, tile size 16, RTX 5070 Laptop.

| mode | intersections | one forward+backward wall sample |
| --- | ---: | ---: |
| AABB OFF | 42,229,706 | 700.09 ms |
| AccuTile ON | 16,998,835 | 599.70 ms |

Room intersection reduction is 59.75%.  This is a single post-build probe, not a
valid 20-warmup/100-measurement benchmark.  Stage-decomposed sort/forward/backward
timings and all bicycle/garden timings are therefore `null`.

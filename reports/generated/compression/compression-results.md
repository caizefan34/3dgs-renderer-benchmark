# EPIC-05 compression matrix

All rows use gsplat packed on decoded standard PLY files. Compressed rows remain
pending until their visual audit is recorded.

| Case | Codec | Ratio | Decode ms | FPS | PSNR | PSNR delta | LPIPS delta | Numeric gate | Visual audit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| small-garden-1080p | reference-ply | 1.000x | 0.0 | 172.11 | 25.827 | +0.000 | +0.0000 | pass | not_applicable |
| small-garden-1080p | block-float | 2.146x | 7408.2 | 172.11 | 25.826 | -0.001 | +0.0002 | pass | pending |
| small-garden-1080p | tile-codebook | 3.803x | 8440.9 | 174.84 | 25.827 | -0.000 | +0.0001 | pass | pending |
| medium-truck-1080p | reference-ply | 1.000x | 0.0 | 244.72 | 24.136 | +0.000 | +0.0000 | pass | not_applicable |
| medium-truck-1080p | block-float | 2.196x | 3658.3 | 244.39 | 24.128 | -0.008 | +0.0010 | pass | pending |
| medium-truck-1080p | tile-codebook | 3.856x | 3753.8 | 249.32 | 24.136 | -0.000 | +0.0001 | pass | pending |
| medium-train-1080p | reference-ply | 1.000x | 0.0 | 266.33 | 22.381 | +0.000 | +0.0000 | pass | not_applicable |
| medium-train-1080p | block-float | 2.219x | 1361.9 | 268.02 | 22.355 | -0.026 | +0.0028 | fail | pending |
| medium-train-1080p | tile-codebook | 3.924x | 1409.5 | 268.50 | 22.380 | -0.001 | +0.0001 | pass | pending |
| large-bicycle-1080p | reference-ply | 1.000x | 0.0 | 188.68 | 24.313 | +0.000 | +0.0000 | pass | not_applicable |
| large-bicycle-1080p | block-float | 2.156x | 8866.2 | 189.08 | 24.309 | -0.004 | +0.0010 | pass | pending |
| large-bicycle-1080p | tile-codebook | 3.845x | 9813.0 | 190.84 | 24.312 | -0.001 | +0.0001 | pass | pending |
| large-bonsai-1080p | reference-ply | 1.000x | 0.0 | 377.29 | 32.513 | +0.000 | +0.0000 | pass | not_applicable |
| large-bonsai-1080p | block-float | 2.267x | 1675.7 | 365.98 | 32.502 | -0.011 | +0.0001 | pass | pending |
| large-bonsai-1080p | tile-codebook | 3.890x | 1722.4 | 256.14 | 32.511 | -0.002 | +0.0001 | pass | pending |

# Measured compression track

This directory contains EPIC-05 compression evidence that is intentionally
kept separate from the common-representation renderer leaderboard.

- `artifact-encoding-2026-07-23.json` measures compressed bytes and CPU
  encode/decode time for all five canonical checkpoints.
- These rows are `artifact_ready`, not near-lossless claims. Rendering FPS,
  PSNR/SSIM/LPIPS, and visual audit remain required before promotion to a
  complete compression result.
- Large ZIP artifacts and decoded PLY files remain on EPIC-05; only hashes and
  compact measurement evidence are committed.

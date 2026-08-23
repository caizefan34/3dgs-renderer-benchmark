# Measured compression track

This directory contains EPIC-05 compression evidence that is intentionally
kept separate from the common-representation renderer leaderboard.

- `artifact-encoding-2026-07-23.json` measures compressed bytes and CPU
  encode/decode time for all five canonical checkpoints.
- The `spz/` rows contain complete decoded-render measurements for SPZ v4 with
  8/8-bit SH. All five numeric and visual gates pass; see
  `reports/epic05-spz-qualification-2026-07-24.md`.
- The original block-float and tile-codebook encoding summary remains
  `artifact_ready`; their per-scene result rows record decoded-render status.
- Large ZIP/SPZ artifacts, contact sheets, and decoded PLY files remain on
  EPIC-05; only hashes and
  compact measurement evidence are committed.

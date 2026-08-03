# Artifacts

Tracked content (committed to git):
- `environment-setup/` — machine reports, CUDA build parameters, `sitecustomize.py`.
- `training/` — training run configs and per-scene assets (cameras, exposure, cfg).

Git-ignored content (not in git; see `.gitignore`):
- `renderer-sources/` — the checked-out third-party renderer source trees,
  including the authoritative HiGS gsplat working tree used to generate
  `patches/higs-differentiable.patch`.

Recovery/reproducibility:
- `scripts/linux/rebuild_higs_csrc.py` rebuilds both CUDA extensions in place.
- A full source snapshot is archived on shared storage:
  `state/backups/gsplat-higs-source-2026-08-02.tar.gz` (see Round 30 in
  `reports/higs-trainability-implementation.md`).

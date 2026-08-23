# Benchmark Matrix v2

This directory is the immutable definition of comparable work.
`suite.json` identifies required cases, `protocol.json` defines measurement boundaries, and `schemas/result.schema.json` defines one result record per renderer configuration and case.

The primary track is `common_representation`: every renderer receives the same checkpoint, Gaussian count, SH degree, camera order, reference images, resolution, and image conventions.
Renderer-native pruning, training, or approximation belongs in the separate
`native_training` track defined by `training.json` and never shares a ranking
with the primary track.

`higs-paper-protocol.json` is stricter than the existing fixed-budget training
matrix. It defines the from-scratch, 11-scene, three-seed, official-baseline and
cross-hardware experiment required for the HiGS method paper. The short-horizon
`run_higs_train_benchmark.py` loads an existing `point_cloud.ply` and therefore
cannot produce a full-training paper row.

`run_higs_full_training.py` is the separate paper launcher. Commands must be
created through `src/scripts/build_higs_training_command.py`, which audits the
local source tree and freezes method, scene, seed, SfM initialization, and the
30k budget. The launcher currently executes the official gsplat control only;
HiGS remains fail-closed until native backward supplies the densification
signals required by `DefaultStrategy`.

Named small/medium/large labels are workload tiers for this suite version, not universal claims about a dataset.
Changing a checkpoint or its Gaussian count requires a new suite version.

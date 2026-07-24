# Benchmark Matrix v2

This directory is the immutable definition of comparable work.
`suite.json` identifies required cases, `protocol.json` defines measurement boundaries, and `schemas/result.schema.json` defines one result record per renderer configuration and case.

The primary track is `common_representation`: every renderer receives the same checkpoint, Gaussian count, SH degree, camera order, reference images, resolution, and image conventions.
Renderer-native pruning, training, or approximation belongs in the separate
`native_training` track defined by `training.json` and never shares a ranking
with the primary track.

Named small/medium/large labels are workload tiers for this suite version, not universal claims about a dataset.
Changing a checkpoint or its Gaussian count requires a new suite version.

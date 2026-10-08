# REMOTE ONLY GUARD

EPIC-05 ONLY.

The user's local Windows PC is control-plane only.

Never run:
- CUDA
- training
- benchmark
- profiling
- large numerical replay

Before execution record:
hostname, GPU, working directory.

Unknown execution host:
STOP_REMOTE_GUARD.

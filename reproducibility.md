# Reproducibility

## Checklist

- Record benchmark suite version.
- Record repository commit hash.
- Record renderer source URL, version, and commit hash when available.
- Record GPU name, VRAM, driver, CUDA runtime/toolkit, PyTorch version, OS, and
  whether clocks were locked.
- Record scene file hash, camera path hash, GT manifest hash, resolution,
  warmup frames, measured frames, repeats, and timing method.
- Keep synthetic stress, quality verification, real-scene speed, and Pareto
  artifacts separate.
- Validate JSON artifacts before publishing.

## Environment Export

```text
python src/scripts/export_environment.py --output results/environment.json
```

## Docker

Renderer environment scaffolds are provided in:

- `docker/gsplat.Dockerfile`
- `docker/higs.Dockerfile`
- `docker/tcgs.Dockerfile`

Example:

```text
docker compose run --build gsplat-benchmark
```

Dockerfiles are intended to make environment setup reproducible. They do not
replace the requirement to publish raw result JSON and metadata.


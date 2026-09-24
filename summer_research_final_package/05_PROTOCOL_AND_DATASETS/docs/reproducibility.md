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

## Known variance: TC-GS tail stalls

The dedicated EPIC-05 re-test on 2026-08-05 used the same A100 GPU UUID,
TC-GS commit, assets, cameras, runner, and 500 measured frames per case as the
published cohort. Three five-scene passes compared the standard 30-frame
warm-up with a 150-frame warm-up.

| pass | warm-up | geometric-mean FPS | scenes with max latency >100 ms |
| --- | ---: | ---: | ---: |
| A | 30 | 217.70 | 5/5 |
| B | 150 | 312.56 | 1/5 |
| C | 30 | 275.46 | 1/5 |

The longer warm-up coincided with fewer stalls but did not eliminate them:
bicycle still reached 338.78 ms in Pass B. Pass C improved under the original
warm-up, so warm-up and changing host load are confounded. Quality is stable:
each scene's PSNR is identical across all passes and historical batches.

Do not infer a kernel defect or host-contention cause from these runs alone.
For a headline TC-GS claim, use an otherwise-idle host, randomized
process-level blocks, at least 10 independent launches per case, and report
median/P95/P99/max latency alongside mean FPS and repeat-level confidence
intervals. Full results and limitations are in
[`reports/epic05-tcgs-variance-2026-08-05.md`](../reports/epic05-tcgs-variance-2026-08-05.md).

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

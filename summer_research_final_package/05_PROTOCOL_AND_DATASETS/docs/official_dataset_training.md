# Official Dataset Training Protocol

Goal: compare globally tracked 3DGS renderers on speed and quality using
official datasets, not self-generated training data.

## Policy

- Training and quality-bearing benchmark submissions must use official dataset
  families used by the original 3DGS evaluation: Mip-NeRF 360, Tanks and
  Temples, and Deep Blending.
- Repository-generated scenes remain available only as the Synthetic Stress
  Suite. They must not enter GT-quality leaderboards.
- Official train/test splits must be used when available. For original 3DGS
  training, use `--eval`.
- Every submission must report dataset family, scene id, camera split,
  resolution, trained checkpoint path, scene hash, and GT manifest hash.

The machine-readable policy is
[`data/datasets/official_training_datasets.json`](../data/datasets/official_training_datasets.json).

## List Official Sources

```text
python src/scripts/download_datasets.py --list-official
```

## Validate Policy

```text
python src/scripts/validate_official_training.py
```

## Training Job Templates

The manifest stores command templates such as:

```text
python train.py -s data/official/mipnerf360/garden -m outputs/official/mipnerf360/garden --eval
```

Training is intentionally not automated in CI because the official datasets
are large and may require explicit license/download steps.

## Migration Notes

Existing historical synthetic timing remains valid as stress-test evidence.
New renderer comparisons that claim quality preservation should be trained and
validated on official scenes before entering quality, balanced, or Pareto
leaderboards.


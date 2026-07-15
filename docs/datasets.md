# Standard Dataset Suite

Suite v3.1.0 fixes five 1920x1080 cases:

| Workload tier | Dataset | Scene | Required |
| --- | --- | --- | --- |
| Small | Mip-NeRF 360 | Garden | Yes |
| Medium | Tanks and Temples | Truck | Yes |
| Medium | Tanks and Temples | Train | Yes |
| Large | Mip-NeRF 360 | Bicycle | Yes |
| Large | Mip-NeRF 360 | Bonsai | Yes |

The workload labels belong to this suite version; they are not universal dataset-size claims.

## Verified sources

Mip-NeRF 360 uses official per-scene GCS objects pinned by generation, size, MD5, and CRC32C. Preparation verifies all four published fields and records the complete local archive SHA-256.

Truck and Train use Graphdeco's official `tandt_db.zip` bundle. The manifest pins its 682,628,995-byte size and SHA-256 `816e62f22a161abbfe841d2a6b10cdf036e297c9fa289b3bfeee9c6ec526d7e1`.

All five checkpoints come from Graphdeco's official pretrained `models.zip`. Each manifest pins the relevant iteration-30000 PLY and `cameras.json` member path, uncompressed size, ZIP CRC32, and source camera SHA-256. Suite v3.1.0 additionally pins the complete checkpoint, derived camera, and GT manifest SHA-256 for every case.

## Preparation

```text
benchmark prepare mipnerf360 --scene garden
benchmark prepare tanks_and_temples --archive PATH_TO_TANDT_DB_ZIP
benchmark prepare-case small-garden-1080p
```

`prepare` downloads or accepts only the manifest source, verifies it before extraction, rejects path traversal, atomically extracts one scene, and writes `datasets/raw/<dataset>/<scene>/dataset_inventory.json`.

`prepare-case` reads the official pretrained model archive remotely by ZIP Range requests, or locally with `--model-archive PATH`. It then:

1. orders official cameras by case-insensitive image name;
2. selects 100 evenly spaced views including both endpoints;
3. center-crops the field of view to 16:9 and maps intrinsics to 1920x1080;
4. records the matching crop for each unmodified source photograph;
5. hashes the PLY, canonical camera JSON, and ordered GT file manifest;
6. stages the case only when all three hashes match `benchmark/suite.json`.

The metric path applies the recorded crop and area-resizes GT at evaluation time. Original photos remain byte-identical, so their manifest hashes do not depend on an image encoder.

`benchmark prepare-case CASE --audit-only` selectively recomputes canonical candidates from official remote ZIP members without downloading whole source archives. It is a maintainer audit tool only: it writes no processed case and cannot authorize Tier A collection.

For reviewed non-canonical experiments, `benchmark stage-case` retains the manual checkpoint/camera/GT path. Such files still cannot enter the primary suite unless their hashes equal the pinned canonical assets.

## Canonical layout

```text
datasets/processed/<dataset>/<scene>/
  point_cloud.ply
  eval_cameras.json
  eval_images/
  preparation.json
```

`preparation.json` records source archive identity, official model members, selected views, conversion rule, all asset hashes, tool commit, and timestamp. Every renderer in the primary track receives these exact files.

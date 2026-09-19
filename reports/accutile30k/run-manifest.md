# Run Manifest — B1A 13-Scene 30K Full-Training Validation

Generated: 2026-09-19T08:43:56.552493+00:00

## Run directories

All runs output to `/dev/shm/accutile30k/<scene>/<method>/` on host `mx`.
Each run directory contains: `config.json`, `provenance.json`, `training_metrics.json`, `evaluation_metrics.json`, `camera_sequence.npy`, `training_results.json`, `MANIFEST.sha256`, `checkpoints/iter_30000.pt`.

| Scene | Dataset | B1 run_id | B1A run_id | Status |
|-------|---------|-----------|-----------|--------|
| bicycle | Mip-NeRF360 | bicycle_b1_s42 | bicycle_b1a_s42 | complete |
| bonsai | Mip-NeRF360 | bonsai_b1_s42 | bonsai_b1a_s42 | complete |
| counter | Mip-NeRF360 | counter_b1_s42 | counter_b1a_s42 | complete |
| flowers | Mip-NeRF360 | flowers_b1_s42 | flowers_b1a_s42 | complete |
| garden | Mip-NeRF360 | garden_b1_s42 | garden_b1a_s42 | complete |
| kitchen | Mip-NeRF360 | kitchen_b1_s42 | kitchen_b1a_s42 | complete |
| room | Mip-NeRF360 | room_b1_s42 | room_b1a_s42 | complete |
| stump | Mip-NeRF360 | stump_b1_s42 | stump_b1a_s42 | complete |
| treehill | Mip-NeRF360 | treehill_b1_s42 | treehill_b1a_s42 | complete |
| train | Tanks & Temples | train_b1_s42 | train_b1a_s42 | complete |
| truck | Tanks & Temples | truck_b1_s42 | truck_b1a_s42 | complete |
| drjohnson | Deep Blending | drjohnson_b1_s42 | drjohnson_b1a_s42 | complete |
| playroom | Deep Blending | playroom_b1_s42 | playroom_b1a_s42 | complete |

## Run IDs

Run IDs are deterministic: `<scene>_<method>_s42` where s42 = seed 42.
Immutable: a re-run would create a new run ID with a timestamp suffix.

## Hardware cohort

| Item | Value |
|------|-------|
| Host | mx (bms-39468022-001) |
| GPU | 8 × NVIDIA A100-PCIE-40GB |
| GPUs used | 0,1,2,3,5,6,7 (GPU 4 excluded — in use by another user) |
| Driver / CUDA | 595.71.05 / CUDA 13.2 |
| Conda env | anysplat |
| PyTorch | 2.4.1+cu124 |

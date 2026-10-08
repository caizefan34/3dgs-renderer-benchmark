# SAD-GS author room 3K protocol (frozen 2026-09-27)

## Scope and interpretation

One full author-configured room training run at the pinned SAD-GS checkout, followed by the authors' held-out render and metric scripts. This is an author-configuration calibration on Python 3.11 / Torch 2.8 / CUDA 12.8; it is not a claim that the result reproduces the paper's dependency stack or an original-loss/common-protocol comparison. The author defaults are retained, including `lambda_dssim=0.2`, `lambda_l2=2.0`, black background, and `resolution=-1`. The author configuration adds L2 to the L1+SSIM objective; report this loss identity as configured and do not relabel it as original-loss SAD-GS.

## Frozen identities

- Source: `/root/codex_sadgs_baseline_20260927`, HEAD `029814fb5ed773e4a794cec5115951a64f610ab8`, clean before run; tracked source tree archive SHA256 `6f67c7928b20e7eecab2f7102c318fad999ac923e56b00ec6d335a7d9ec9efde`.
- Dataset: `/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room`; 311 COLMAP camera images. `images.sha256` contains SHA256 for each effective input image; `colmap.sha256` hashes the COLMAP sparse inputs; `split.json` freezes camera ordering and exact split names. The author loader sorts `image_name`, then holds out indices divisible by 8: 272 train, 39 test.
- Runtime: Python 3.11.16, Torch 2.8.0+cu128, toolkit CUDA 12.8; clone environment plus read-only Torch site-packages path. Native extension SHA256 values and key source file hashes are in `frozen_inputs.json`, `source.sha256`, and this run's copied manifest.
- Device: physical A100-SXM4-80GB index 3, UUID `GPU-431ac351-20f8-bdf7-d798-1c9a3f545b72`. GPU2 was occupied, so the idle, same-model GPU3 was selected. Pre-start requirement: selected GPU UUID matches, used memory <=256 MiB, utilization <=10%, and no compute PID. No allocator cap is applied. Monitor once per second; stop if total GPU memory reaches 73,728 MiB or a single phase exceeds 21,600 seconds. This preserves at least 8 GiB of device memory under the memory stop gate. GPU compute PIDs are logged, but not used for ownership decisions because EPIC-05 host/container PID namespaces differ; the supervisor only terminates the process group it launched.

## Exact commands

Working directory: `/root/codex_sadgs_baseline_20260927`. All commands use the clone Python executable and `PYTHONPATH=/root/miniforge3/envs/sadgs_baseline_clone_20260927/lib/python3.11/site-packages:/root/miniforge3/envs/codex_nerficg/lib/python3.11/site-packages`, `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=3`, `CUDA_HOME=/usr/local/cuda-12.8`, and `TORCH_CUDA_ARCH_LIST=8.0`. `OAR_JOB_ID=room` is set on the train command.

```bash
/root/miniforge3/envs/sadgs_baseline_clone_20260927/bin/python train.py -s /root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room -i images --eval --scale_rotation_scheduler --adam_eps_order 10 --freq_transmittance_threshold 0.8 --freq_opacity_threshold 0.5 --batch_size 2 --optimizer_type hybrid --sample_bbox_faces --warmup_densification --mult 0.25 --iterations 3000 --position_lr_max_steps 3000 --opacity_reset_interval 1200 --densification_interval 500 --densify_until_iter 1900 --densify_from_iter 100 --save_iterations 3000 --prune_iterations 1900 --model_path /root/sadgs_room_author_3k_20260927_attempt2/model
/root/miniforge3/envs/sadgs_baseline_clone_20260927/bin/python render.py -m /root/sadgs_room_author_3k_20260927_attempt2/model --skip_train --iteration 3000
/root/miniforge3/envs/sadgs_baseline_clone_20260927/bin/python metrics.py -m /root/sadgs_room_author_3k_20260927_attempt2/model
```

The training command preserves the indoor Config 2 flags from `run_train.sh`; `--model_path` is redirected to an isolated output directory. `--resolution` is omitted as in the author's command; the source default is `-1` and loader cap is width 1600. Training loop CUDA-event sum (`Training time`) is recorded separately from the supervisor's complete process wall (train, test render, metrics). Metrics must contain per-view PSNR/SSIM/LPIPS for all 39 held-out views and match the frozen split names. Logs, 1-second GPU memory samples, phase timing/exit status, source identity before/after, and output PLY hash are retained in this run directory.

Attempt 001 startup abort evidence is preserved in `attempt_001.md`, `attempt_001_gpu_samples.csv`, `attempt_001_train.log`, and `attempt_001_supervisor_exception.json` in the original run directory. Attempt 002 uses a clean output path.

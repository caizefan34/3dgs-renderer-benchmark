# SAD-GS room 5K original-loss frontier gate (2026-09-27)

## Scope and claim boundary

One author-code run from scratch at 5,000 iterations on Mip-NeRF 360 room, using the previously verified L1+SSIM objective (`lambda_l2=0`). All indoor Config 2 training settings remain fixed except `iterations` and `save_iterations`, which change from 3,000 to 5,000. In particular, `position_lr_max_steps=3000`, densification/pruning topology schedule, batch size, optimizer, scale/rotation scheduler, `mult=0.25`, data split, 1600-wide loader, device, and official evaluation flow remain unchanged. This is a single-scene, single-seed upper-bound calibration of an existing method, not an original method or a multisccene/SOTA claim. No 7K run follows this gate.

## Loss formula

The loss-parity audit is in `LOSS_PARITY.md`. `lambda_l2=0` yields the same scalar objective as the inspected FastGS source: `0.8*mean(abs(render-GT)) + 0.2*(1-fused_ssim)`. This run tests extra training horizon only. SAD-GS still uses a different renderer, densification and optimization system; comparisons do not establish full FastGS same-protocol superiority.

## Frozen identities and comparisons

- Source `/root/codex_sadgs_baseline_20260927`, clean HEAD `029814fb5ed773e4a794cec5115951a64f610ab8`, archive SHA256 `6f67c7928b20e7eecab2f7102c318fad999ac923e56b00ec6d335a7d9ec9efde`.
- Dataset `/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room`; exact image, COLMAP and split hashes in adjacent manifests. Loader order yields 272 train / 39 held-out test views; effective width 1600 from author default `resolution=-1`.
- Runtime Python 3.11.16, Torch 2.8.0+cu128, CUDA 12.8. Device physical A100-SXM4-80GB GPU3 UUID `GPU-431ac351-20f8-bdf7-d798-1c9a3f545b72`, same as SAD-GS 3K arms and FastGS-Big run host. No allocator cap. Startup gate: exact UUID, used memory <=256 MiB, utilization <=10%, no compute PIDs. Monitor once per second and terminate only this supervisor's process group if device memory reaches 73,728 MiB or a phase exceeds 21,600 s; host/container PID numbers are not used to attribute compute processes.
- Paired SAD-GS original-loss 3K-from-scratch result: PSNR 31.360905 dB, SSIM .930564, LPIPS .180577; train-process wall 140.937 s, author CUDA-event training sum 72.084830 s, full train→render→metrics 307.133 s. Frozen report is `SAD_GS_ROOM_L2ZERO_3K_20260927/RESULTS.md` in the local project.
- FastGS-Big from-scratch 30K reference supplied by the fixed baseline ledger: complete wall 287.27 s, PSNR 32.20546 dB, SSIM .930243, LPIPS .188115. Use as an existing-method context only; SAD-GS and FastGS remain different systems and this one-scene comparison is not a complete matched strong-baseline study.

## Exact commands

Working directory `/root/codex_sadgs_baseline_20260927`. Environment sets `CUDA_VISIBLE_DEVICES=3`, `CUDA_HOME=/usr/local/cuda-12.8`, `TORCH_CUDA_ARCH_LIST=8.0`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH` to clone extensions plus read-only Torch site-packages. `OAR_JOB_ID=room` for training. The only config changes from the SAD-GS original-loss 3K arm are `--iterations 5000` and `--save_iterations 5000`; `--position_lr_max_steps 3000` is deliberately retained.

```bash
/root/miniforge3/envs/sadgs_baseline_clone_20260927/bin/python train.py -s /root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room -i images --eval --scale_rotation_scheduler --adam_eps_order 10 --freq_transmittance_threshold 0.8 --freq_opacity_threshold 0.5 --batch_size 2 --optimizer_type hybrid --sample_bbox_faces --warmup_densification --mult 0.25 --iterations 5000 --position_lr_max_steps 3000 --opacity_reset_interval 1200 --densification_interval 500 --densify_until_iter 1900 --densify_from_iter 100 --save_iterations 5000 --prune_iterations 1900 --lambda_l2 0.0 --model_path /root/sadgs_room_author_5k_l2zero_20260927/model
/root/miniforge3/envs/sadgs_baseline_clone_20260927/bin/python render.py -m /root/sadgs_room_author_5k_l2zero_20260927/model --skip_train --iteration 5000
/root/miniforge3/envs/sadgs_baseline_clone_20260927/bin/python metrics.py -m /root/sadgs_room_author_5k_l2zero_20260927/model
```

Track training-process wall separately from the author's CUDA-event `Training time`, plus complete train→render→metrics wall and 39 per-view PSNR/SSIM/LPIPS. The validator checks `00000.png`…`00038.png`, matching render/GT filename sets, JSON metric coverage, source/data state and final PLY hash. Source, code and data hashes are frozen before launch.

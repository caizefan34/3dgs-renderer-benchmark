# SAD-GS room 3K loss-matched author baseline (2026-09-27)

## Question and interpretation

Run the pinned SAD-GS room indoor Config 2 for 3,000 steps with only `lambda_l2=0.0`, then use the same official held-out rendering and metric path as the paired SAD-GS `lambda_l2=2.0` author-config run. The purpose is to isolate the contribution of SAD-GS's extra L2 term under this author training recipe. It is not an original-method intervention, a multi-scene win, or a full-protocol FastGS comparison; renderer, densification, optimizer, and other method settings still differ from FastGS.

## Loss definition gate (passed before launch)

Pinned SAD-GS `train.py` computes `L = (1-lambda_dssim)*Ll1 + lambda_dssim*(1-fused_ssim) + lambda_l2*Ll2`, with `Ll1=mean(abs(render-GT))`, `Ll2=mean((render-GT)^2)`, and `lambda_dssim=0.2`. This arm passes `--lambda_l2 0.0`, so the optimized objective is exactly `0.8*mean(abs(render-GT)) + 0.2*(1-fused_ssim)`.

The local FastGS training implementation uses the same L1/D-SSIM scalar equation and the same L1 mean. FastGS and SAD-GS `fused-ssim` `setup.py`, `ssim.cu`, `ext.cpp`, and `fused_ssim/__init__.py` source text is identical after normalizing Windows CRLF versus Linux LF. FastGS reference hashes and source-line evidence are recorded in `LOSS_PARITY.md`. Therefore the extra L2 term is removed without changing the L1/SSIM definition. This establishes loss-expression parity only, not full-system or all-training-protocol parity.

## Frozen identities and paired arm

- SAD-GS source: `/root/codex_sadgs_baseline_20260927`, detached HEAD `029814fb5ed773e4a794cec5115951a64f610ab8`, clean before run. Full git archive hash and key file hashes are in `frozen_inputs.json` / `source.sha256`.
- Dataset: `/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room`. Frozen per-image checksums and COLMAP/split manifests are in `images.sha256`, `colmap.sha256`, and `split.json`. The loader's sorted camera order with every eighth image held out gives 272 train and 39 test views. `resolution=-1` remains the author default and width cap is 1600.
- Runtime: Python 3.11.16, Torch 2.8.0+cu128, CUDA 12.8; same isolated prefix and read-only Torch site-packages used for the paired L2=2 arm. No CUDA allocator cap.
- Device: physical A100-SXM4-80GB index 3, UUID `GPU-431ac351-20f8-bdf7-d798-1c9a3f545b72`, same as paired L2=2 arm. Pre-start gate: UUID matches, used memory <=256 MiB, utilization <=10%, no compute PIDs. Sample GPU memory/utilization once per second, stop if total use reaches 73,728 MiB or a phase exceeds 21,600 s. Host/container PID attribution is not used at runtime; the supervisor only terminates its own launched process group.
- Paired L2=2.0 arm result: PSNR 31.435127 dB, SSIM 0.930348, LPIPS 0.181309; train process wall 147.618 s, author CUDA-event sum 76.751473 s, full train→render→metrics 314.032 s, sampled GPU peak 31,729 MiB. Its full record is `SAD_GS_ROOM_AUTHOR_3K_20260927/RESULTS.md` in the local project.

## Exact commands

Working directory `/root/codex_sadgs_baseline_20260927`, clone Python and runtime environment as above, `CUDA_VISIBLE_DEVICES=3`, `PYTHONDONTWRITEBYTECODE=1`, `OAR_JOB_ID=room` for training. All settings copied from indoor Config 2; the only recipe change is `--lambda_l2 0.0`. `--model_path` isolates the output. Author `render.py --skip_train --iteration 3000` and `metrics.py` are unchanged from the paired arm.

```bash
/root/miniforge3/envs/sadgs_baseline_clone_20260927/bin/python train.py -s /root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room -i images --eval --scale_rotation_scheduler --adam_eps_order 10 --freq_transmittance_threshold 0.8 --freq_opacity_threshold 0.5 --batch_size 2 --optimizer_type hybrid --sample_bbox_faces --warmup_densification --mult 0.25 --iterations 3000 --position_lr_max_steps 3000 --opacity_reset_interval 1200 --densification_interval 500 --densify_until_iter 1900 --densify_from_iter 100 --save_iterations 3000 --prune_iterations 1900 --lambda_l2 0.0 --model_path /root/sadgs_room_author_3k_l2zero_20260927/model
/root/miniforge3/envs/sadgs_baseline_clone_20260927/bin/python render.py -m /root/sadgs_room_author_3k_l2zero_20260927/model --skip_train --iteration 3000
/root/miniforge3/envs/sadgs_baseline_clone_20260927/bin/python metrics.py -m /root/sadgs_room_author_3k_l2zero_20260927/model
```

The training-process wall (including initialization/precompute/save) is recorded separately from the author's CUDA-event `Training time`. Complete train→render→metrics process wall, stage exit codes, one-second GPU memory samples, image metrics/per-view validation, source state and final PLY hash are retained under this run directory.

# SAD-GS bounded CUDA smoke on EPIC-05 (2026-09-27)

This validates that pinned author code can initialize room and execute its native CUDA training path on the available A100. It is **not** a complete author reproduction, a comparable baseline time, a quality result, or an original method experiment. See `SAD_GS_BASELINE_PREP_20260927.md` for source, extension and dataset identities.

## Frozen scope

Author checkout `/root/codex_sadgs_baseline_20260927` at detached HEAD `029814fb5ed773e4a794cec5115951a64f610ab8`; clone extension prefix and read-only Torch 2.8/cu128 source described in the preparation record. Both attempts used GPU2 (`GPU-bcca92b9-0234-7ec9-a2f0-a63925d52146`), room dataset, `--eval`, author room-indoor options from `run_train.sh` with `--iterations 2` and isolated `--model_path`, 240-second process timeout. Effective `-1` camera-loader scaling reported 1600×1066. The only deliberate non-author smoke controls were the short iteration bound and process allocator fraction. The training loop labels 0,1,2; optimizer executes labels 0 and 1. No final PSNR/SSIM/LPIPS was measured.

## First attempt: artificial memory gate failed before training

Directory `/root/sadgs_room_cuda_smoke_20260927`; `run.sh` / `sitecustomize.py` SHA256 `7b8595f2b74df495f83d0696aa49d725f96723029da759d1839564ea27179550` / `7f8fbdb7bbda579180858dfd05cf91c748c826a0cf3a5070528ef95137087f99`. The allocator cap was 0.25 of A100 memory (about 20 GiB). Exit code 1 after 53.289 seconds complete process wall. The author pre-loop multiscale structure-tensor precomputation for 272 room training views, batching up to 100 images at 1600×1066, hit `torch.OutOfMemoryError` requesting 652 MiB with 19.46 GiB already allocated. It did **not** reach a training iteration. `run.log` / `run.meta` SHA256 `65bca16006e28ddd97493d370fe9034e3b4af5abe0d947c6ede225a265004644` / `a00d319a6df7bd2651b7290546df029e2344a08841b94a0e3f4fef19e1abedec`. This is a failure of the imposed 20 GiB cap, not evidence that an 80 GiB author run fails.

## Single bounded retry: native path completed

Directory `/root/sadgs_room_cuda_smoke_retry_20260927`; `run.sh` / `sitecustomize.py` SHA256 `e01e80d228bdad11f591590693f6b1885c33aec292519f4c48c9986734b6d65a` / `03fb5d1a691b5179854c2e2c17603f4a8a40cdb4d06cb58288ba411a11e1d407`. The only changed runtime control was allocator cap 0.75 (about 60 GiB); `PYTHONDONTWRITEBYTECODE=1` was added so the author checkout would not rewrite tracked `.pyc` files. Exit code 0, process wall 57.839 seconds. Author log reports 272 training cameras at 1600×1066, 112,627 initial points plus 1,536 bounding-box-face points, structure-tensor precompute 1.1249 seconds, `Training time: 0.6930312337` seconds (sum of author CUDA event intervals), 114,163 final Gaussians, and `Training complete.` Peak Torch allocated/reserved were 29,767,603,712 / 32,692,502,528 bytes. The saved iteration-2 PLY is 28,770,632 bytes, SHA256 `0e25674b704a37cb1f605329e43459f6ea383333f709b5b5f901707e680f3f12`. `run.log` / `run.meta` SHA256 `daeff701aa97d5004227db99ca05a7a9d6136bee6613a1cee11b0c56818bd23a` / `cafd465d96292d55a9a0f5032cb5e03d015878865229389c003c1bacc986e9ed`.

The log does not print an explicit loss value in this short interval, so finite loss was **not independently verified**; successful backward/optimizer and PLY save establish only end-to-end execution. No image-quality or numerical parity claim follows. The 57.8 versus 0.693 second distinction is dominated by startup on a two-step process and cannot be extrapolated to 3K timing.

## Post-run state

Both SSH execution sessions reached terminal exit; GPU2 returned to 0 MiB/0% with no compute process on its UUID. The first attempt modified tracked author `.pyc` files; these 20 bytecode paths were restored from the pinned checkout, and its lone generated top-level `__pycache__` removed after path/content validation. Final author Git status is clean and HEAD unchanged. Native imports from both the original `codex_nerficg` environment and the SAD-GS clone prefix were rechecked successfully. No full author run, formal held-out evaluation, or original-loss/complete-wall comparison has begun.

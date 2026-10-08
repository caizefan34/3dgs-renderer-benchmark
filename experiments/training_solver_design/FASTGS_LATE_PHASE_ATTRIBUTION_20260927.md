# Late FastGS training: quality evidence and attribution gap

Checked 2026-09-27 from the existing paired room trajectory in `FASTGS_BARRIER_AUDIT.md`, not a new run. Both branches started from the same optimizer-complete 4,099-step checkpoint; this page uses only the **unchanged author-score branch**. It trained at effective 1600×1066 resolution with the original loss and evaluated the same 39 official retained room views at 10K/20K/30K. Those views have been used for development and are not an untouched confirmation set.

| Iteration | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---:|---:|---:|---:|
| 10K | 30.400068 | .9136015 | .2174812 |
| 20K | 32.020439 | .9291240 | .1912480 |
| 30K | 32.229950 | .9302623 | .1887598 |

Thus 20K→30K changed the held-out aggregate by **+0.209511 dB PSNR, +.0011383 SSIM, −.0024882 LPIPS**. These are within one room trajectory, not a multi-scene variance estimate. They contradict the simple claim that deleting the final 10K steps would preserve this run's endpoint quality. They do not show whether later improvements came from geometry/opacity, appearance, pruning, or their interaction. The native late-stage profile (`FASTGS_STAGE_PROFILE_RESULTS.md`) shows raster backward is 45–53% of active CUDA time in room/garden windows, but cost attribution cannot identify which parameter updates caused the held-out improvement.

The next diagnostic question is whether late updates correct reusable geometry or fit view-dependent appearance/capture inconsistency. A fair selective-update continuation would need the *same* 20K model **and Adam state**, identical train-camera order, original full-resolution loss and original topology schedule, with paired unchanged and selectively frozen arms; otherwise optimization history and topology confound the result. It would still be a diagnostic, not an original quality-and-speed algorithm. No such continuation was run; first verify that the required checkpoint exists and that a distinct intervention could use the answer. Do not spend an untouched scene or call a short pilot final evidence.

## Artifact feasibility check (read-only)

Luna checked the pinned author repository at `/root/codex_fastgs_audit_20260927`, source commit `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`, `train.py` SHA256 `1fd700c341d109f13f95ff3a3cdf1328445f27909119e7eacadb31913e34c963`. Root then verified in the same source that `checkpoint_iterations` is accepted as an argument (default `[30000]`) but **never used inside the training loop to write a checkpoint**; the earlier 4,099 checkpoint required the separate `fastgs_capture_checkpoint.py` wrapper. Its current `output` tree has no room or garden **20K optimizer-complete checkpoint**, and no garden 20K PLY. The read-only search found three room 20K PLY-only snapshots:

| Remote path relative to `output/` | Bytes | SHA256 |
|---|---:|---|
| `room_big_branch_base30k/point_cloud/iteration_20000/point_cloud.ply` | 144,394,075 | `afa745627f7e7dd0fe1606574c3ed48d660afba1375e6686e862f5d42ef02fcb` |
| `room_big_branch_cap32_30k/point_cloud/iteration_20000/point_cloud.ply` | 144,293,883 | `142a5ef5e9888866ce49b421a5ed356576214d1aa52a1b7f029807e4932ea9bb` |
| `room_pose_order_30k/point_cloud/iteration_20000/point_cloud.ply` | 90,559,227 | `63391d59a62e4d5c583e21fb04893f4689fdd06b93dca11637ae61d1ac22c971` |

The first PLY's `cfg_args` matches room, the author's image input, effective source-resolution setting and `eval=True`; none stores Adam moments. Using any of them as a new 20K optimizer-continuation origin would reset the trajectory and invalidate the proposed matched-history attribution. A new paired 20K checkpoint would require rerunning from an earlier optimizer-complete state with an explicit capture wrapper; passing `--checkpoint_iterations 20000` alone would not save it. A model/Adam capture also omits Python RNG and the remaining camera stack, so a resumed pair could be internally matched but would not exactly continue the original unbroken camera sequence without extra state capture. No GPU run was launched for this artifact check. The author `eval=True` split is 272 train/39 held-out for room and 161/24 for garden by sorted image-name eighths; the room 39 are already development views, and garden should remain untouched for future confirmation.

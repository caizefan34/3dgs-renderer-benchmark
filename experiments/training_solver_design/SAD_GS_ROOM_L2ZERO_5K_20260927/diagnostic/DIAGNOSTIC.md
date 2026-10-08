# SAD-GS room train-view residual/work diagnostic

Read-only post-hoc diagnostic for the independent original-loss SAD-GS 3K and 5K checkpoints. No training, checkpoint edits, source edits, rasterizer rebuilds, test-view selection, or test-view filtering were performed.

## Frozen selection and execution

Sort the 272 training cameras by `image_name`; select indices `round(i*(272-1)/7)` for `i=0..7`: `0, 39, 77, 116, 155, 194, 232, 271`. The selected camera names are `DSCF4668.JPG`, `DSCF4712.JPG`, `DSCF4756.JPG`, `DSCF4800.JPG`, `DSCF4845.JPG`, `DSCF4889.JPG`, `DSCF4933.JPG`, and `DSCF4977.JPG` (all 1600×1066). This selection was fixed from the training-camera list only and does not use proximity to the held-out outlier.

The 3K and 5K PLYs were loaded through the pinned author `Scene`/`GaussianModel` and `render_structgs` code. The author `render.py` default `--mult` is 0.5; the frozen training command used `--mult 0.25`. Both values were measured and are reported separately. CUDA `duplicateToTilesTouched` uses `mult` to change the opacity-threshold ellipse that generates tile support; the inner compositor's accepted-alpha and early-transmittance thresholds remain unchanged. Therefore 0.5 is the official render setting, while 0.25 matches the training rasterizer setting for this forward-only diagnostic.

For each selected frame, the script rendered at effective input resolution, measured mean absolute RGB train-image residual and its per-pixel p90, then used the rasterizer's existing `get_flag`/`metric_map` path. “Accepted work” means summed Gaussian-pixel pair counts from `accum_metric_counts`: a pair passed the power test, alpha was at least 1/255, and the candidate did not trigger the `test_T < 1e-4` early stop. This is an accepted forward-contribution count, not instruction count, backward work, elapsed time, or total training cost. “High projected footprint” ranks visible Gaussians by `9π√det(cov2D)`, a 3-sigma screen-space ellipse area proxy. A “front-visible translucent Gaussian” has filtered opacity in [0.1, 0.9] and measured maximum preceding transmittance at least 0.1. “Semitransparent pixels” have rendered opacity in [0.1, 0.9]. The renderer's depth map is divided by opacity before foreground depth summaries.

The remote commands were identical except for the checkpoint and `--mult`:

```sh
cd /root/codex_sadgs_baseline_20260927
export CUDA_VISIBLE_DEVICES=3 CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.0 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/root/codex_sadgs_baseline_20260927:/root/miniforge3/envs/sadgs_baseline_clone_20260927/lib/python3.11/site-packages:/root/miniforge3/envs/codex_nerficg/lib/python3.11/site-packages
/root/miniforge3/envs/sadgs_baseline_clone_20260927/bin/python /root/sadgs_room_author_5k_l2zero_20260927/diagnostic/diagnose_train_views.py \
  --model_path MODEL --iteration ITERATION --mult MULT --output OUTPUT.json --quiet
```

The four `(MODEL, ITERATION, MULT, OUTPUT)` settings were `(sadgs_room_author_3k_l2zero_20260927/model, 3000, 0.5, train_views_3k.json)`, `(sadgs_room_author_5k_l2zero_20260927/model, 5000, 0.5, train_views_5k.json)`, and the corresponding `_mult025.json` outputs with `MULT=0.25`.

## Results

Each cell compares mean over the same eight training views. A lower residual is better.

| Rasterizer `mult` | 3K → 5K mean RGB MAE | Views with lower MAE | 3K → 5K mean residual p90 | Accepted work in top 10% residual pixels | Accepted work from top 1% / 10% footprint Gaussians | Accepted work from front-visible translucent Gaussians | Semitransparent pixel area / accepted work | Mean opacity / final transmittance | Mean normalized foreground depth |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 (training setting) | .013158 → .012164 (−.000995) | 7/8 | .029289 → .026821 | 10.51% → 10.58% | 13.90% / 49.89% → 13.88% / 49.63% | 60.18% → 54.81% | 3.25% / 1.02% → 2.61% / 1.10% | .98720 / .01280 → .99159 / .00841 | 5.9007 → 5.8496 |
| 0.5 (author `render.py` default) | .013512 → .012529 (−.000982) | 7/8 | .029895 → .027596 | 10.37% → 10.44% | 18.60% / 56.16% → 18.45% / 55.81% | 59.05% → 53.55% | 2.71% / 1.05% → 1.80% / .91% | .99034 / .00966 → .99394 / .00606 | 5.8814 → 5.8322 |

The top 10% of residual pixels account for roughly 10.4–10.6% of accepted pairs, close to their image-area share. The large-footprint tail carries substantial accepted work, but its share changes little between these two independently trained snapshots. Most output pixels are nearly opaque: median rendered opacity is about .9998, and mean transmittance is below .013. The Gaussian-opacity bucket appears in roughly 54–60% of accepted pairs, but the actual semitransparent pixel region accounts for only about 0.9–1.1% of accepted pairs.

On these registered training views, the 5K render has lower mean residual on 7/8 views and lower average p90 residual at both `mult` settings. Only `DSCF4933` worsens (MAE +.001313 at `mult=.25`, +.001066 at `mult=.5`). This does not reproduce a broad training-view residual regression or show that the held-out PSNR outlier dominates accepted image work. It does not identify the splat(s) behind the held-out veil or its training-view cause. The measured comparison is between independent from-zero runs whose logged topology counts already differ at step 3000, so it cannot isolate duration as a cause.

## Evidence and limits

`diagnose_train_views.py` SHA256 is `35fde229938136c2871d07a4ce839106c7ee1055a64d50331deb1e41c958dbba`. The four JSON SHA256 values are recorded in `SHA256SUMS.txt`; local copies were byte-verified against EPIC-05. Checkpoint, camera, source and compiled-extension hashes, the clean author HEAD, and post-run GPU release are recorded in `REMOTE_IDENTITIES.md`.

This is eight sampled training views from one scene and one independent run at each duration. It cannot establish a scene-wide pattern, backward/optimizer cost, causality, or a fix. Keep the 39 held-out views and official aggregates unchanged. No additional views or training run are warranted by this diagnostic.

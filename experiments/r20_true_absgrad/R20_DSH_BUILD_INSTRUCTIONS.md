# R20 DSH MX build and validation

Status: INSTRUCTIONS_COMPLETE; commands not run on MX by Codex.
Date: 2026-10-09. Target: NVIDIA A100-PCIE-40GB. Use the same existing R18/R19
Python environment and toolchain. Keep the frozen source directory read-only.

## 1. Prepare and build a new candidate

The R20 directory must be available in the MX repository. These commands use
the actual exported MX source path and a separate new candidate/output path.
Select an available GPU through CUDA_VISIBLE_DEVICES; the value below is 0.

```bash
set -euo pipefail
export R20_TASK=/home/liaoyuanjun/3dgs-renderer-benchmark/experiments/r20_true_absgrad
export R20_FROZEN=/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx
export R20_OUT=/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_ABSGRAD_PARITY_R20_MX
export R20_CANDIDATE="$R20_OUT/gsplat-higs-r20"
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=4
mkdir -p "$R20_OUT/logs"
python "$R20_TASK/R20_GRADIENT_PARITY_TESTS.py" --cpu --output "$R20_OUT/cpu.json"
python "$R20_TASK/R20_PREPARE_MX.py" --frozen-root "$R20_FROZEN" --candidate-root "$R20_CANDIDATE"
python "$R20_TASK/R20_BUILD_MX.py" --candidate-root "$R20_CANDIDATE" > "$R20_OUT/logs/build.log" 2>&1
```

Do not reuse an existing candidate directory. Preparation verifies both frozen
binary SHA256 values and all four original source hashes BEFORE copying. Build
uses the actual MX experimental builder with an isolated `.r20_build` directory
and module name `csrc`, so it exports the required `PyInit_csrc`. It copies the
new experimental binary into the NEW candidate package, and never rebuilds
`gsplat/csrc.so`. `R20_PREPARATION.json` records shared header hashes;
`R20_BUILD_IDENTITY.json` records the actual new binary and build flags.
Keep build.log including ptxas register/spill information. A source preimage or
dependency mismatch is a stop, not authorization to relax the hash check.

## 2. Synthetic native/frozen comparison

Each test starts four isolated processes: frozen std_ll native AbsGrad, R20 HiGS
ON, R20 HiGS OFF, frozen HiGS OFF. This avoids mixing registered torch operators
from two roots in one process. All processes verify the frozen main hash;
legacy HiGS also verifies its experimental hash. Candidate requires version 1.

```bash
for px in 0 1 2 4; do
  HIGS_PX_RUNTIME="$px" python "$R20_TASK/R20_GRADIENT_PARITY_TESTS.py" --gpu \
    --reference-root "$R20_FROZEN" --candidate-root "$R20_CANDIDATE" \
    --output "$R20_OUT/parity_px${px}/results.json"
done
```

Cases: same direction, cancellation, single Gaussian, overlap, partial tile
boundary, mixed/all invisible, multiple cameras, alpha-only, alpha cap,
threshold below/above 0.0008, and raw SH3/sigmoid/exp/normalized quaternion chains.
Both forward RGB/alpha, signed gradients, raw parameter gradients and absolute
components are checked. Decision reports include exact threshold, clone/split
and parent-prune masks, mismatch IDs, top-K overlap, MAE/RMSE/max/relative/cosine.
Default numeric gates are allclose(atol=2e-6,rtol=.002), exact decision/radii
agreement and top-1000 overlap >=.99 for large N; adjust tolerances only with
reported reference repeatability evidence. Synthetic-only success explicitly
reports `PASS_SYNTHETIC_REAL_NOT_EXECUTED`, not a complete R20 PASS.

If native HiGS culling itself fails a case, rerun the same command with
`--projection-culling` as a SEPARATE diagnostic and retain both results. A pass
with that switch does not certify the requested HiGS culling configuration.

## 3. Real checkpoint fixture and full-count/B4 parity

Use the actual R18 raw checkpoint (4,413,928 rows) and its recorded SHA256. Obtain
the scene extent from the R19 reference recipe's training cameras; it is the
max pairwise camera-center distance over the first min(50,n_train_cameras),
floored at .1. Export it as R20_SCENE_EXTENT. This value controls clone/split and
world-prune decisions, so do not substitute an estimated training extent.

```bash
export R20_CKPT=/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c/checkpoints/iter_30000.pt
export R20_CAMERAS=/home/liaoyuanjun/higs-13scene/data/official/mipnerf360/bicycle/cameras.json
: "${R20_SCENE_EXTENT:?Set the reference recipe scene extent}"
python "$R20_TASK/R20_MAKE_REAL_FIXTURE.py" \
  --checkpoint "$R20_CKPT" \
  --checkpoint-sha256 ccec425c8da02d423fb66e89450e3c837209f280deea24dbf4a507bc3cd4ea14 \
  --cameras-json "$R20_CAMERAS" --camera-ids 0,1,2,3 --width 960 --height 540 \
  --extent "$R20_SCENE_EXTENT" --output "$R20_OUT/bicycle_full_b4.pt"
HIGS_PX_RUNTIME=2 python "$R20_TASK/R20_GRADIENT_PARITY_TESTS.py" --gpu \
  --reference-root "$R20_FROZEN" --candidate-root "$R20_CANDIDATE" \
  --real-fixture "$R20_OUT/bicycle_full_b4.pt" --output "$R20_OUT/real_b4/results.json"
```

The maker implements the actual R18 xyz/rotation/scaling/opacity/shs schema and
3DGS cameras.json rotation/position/fx/fy/width/height convention, activates
parameters inside the tested graph, and hashes original files before/after.
For a preliminary smaller check use `--subset 580000` in a NEW fixture file;
retain the full-count gate. Repeat real parity at B1 and 1080p with new fixtures.
CPU captures retain all Gaussian gradients and can consume several GB; the
float64 metric calculation adds CPU memory. OOM is FAIL/NOT_EXECUTED, never PASS.
All candidate decisions here use identical fixed parents. Full clone/split child
states and post-prune topology require paired training and fixed random seeds.

## 4. AbsGrad overhead microbench

```bash
for repeat in 1 2 3; do
  for backend in std higs; do
    root="$R20_FROZEN"
    if [ "$backend" = higs ]; then root="$R20_CANDIDATE"; fi
    for mode in off on; do
      args=()
      if [ "$mode" = on ]; then args+=(--absgrad); fi
      HIGS_PX_RUNTIME=2 python "$R20_TASK/R20_ABSGRAD_MICROBENCH.py" \
        --root "$root" --backend "$backend" "${args[@]}" \
        --real-fixture "$R20_OUT/bicycle_full_b4.pt" --warmup 20 --iterations 200 \
        --output "$R20_OUT/bench/${backend}_${mode}_${repeat}.json"
    done
  done
done
```

Repeat with reversed arm order if thermal/co-run drift is suspected; record other
GPU jobs. Save JSON+CSV: forward/backward/full GPU interval, host wall, allocated
and reserved memory, peaks, p90/p99/max/outlier iterations. Binary hashes are
verified before/after. No optimizer/densify or empty_cache occurs in timed work.
Views and cotangents are resident before measurement. Source allocation and
scatter overhead remain part of actual renderer backward. PX=0/1/4 comparisons
use their own reports. Do not infer A100 overhead from the local RTX test.

## 5. Training validation after parity

`R20_TRAINER.py` is a separate copy of R19's trainer with only identity/API/hook
selection updates and explicit eps2d=.1 alignment. Shared reference_v1 model,
loss, LR/SH/camera schedules, optimizer, thresholds and checkpoint stay unchanged.
The false/signed arm's hook is explicitly a historical signed signal; it is never
used for True AbsGrad. Training dynamically changes N with freeze_topology=False.

```bash
for scene in bicycle room; do
  HIGS_PROJECT="$R20_CANDIDATE" python "$R20_TASK/R20_TRAINER.py" \
    --backend higs --scene "$scene" --densify-mode absgrad --iterations 1000 \
    --seed 42 --outdir "$R20_OUT/smoke/${scene}_higs"
done
for scene in bicycle room; do
  HIGS_PROJECT="$R20_FROZEN" python "$R20_TASK/R20_TRAINER.py" \
    --backend std_ll --scene "$scene" --densify-mode absgrad --iterations 30000 \
    --seed 42 --outdir "$R20_OUT/train/${scene}_std"
  HIGS_PROJECT="$R20_CANDIDATE" python "$R20_TASK/R20_TRAINER.py" \
    --backend higs --scene "$scene" --densify-mode absgrad --iterations 30000 \
    --seed 42 --outdir "$R20_OUT/train/${scene}_higs"
done
```

Keep camera_sequence.npy, per-step candidate diagnostics at densification
boundaries, final Gaussian counts, PSNR/SSIM, actual binary identities and raw
timing traces. Compare paired loss/gradient/candidate masks BEFORE topology
divergence; use identical seeds for split-child sampling. Only complete 30K
results can support a training-quality recovery statement. The inherited trainer
phase summaries alone cannot resolve the R19 outlier mechanism; instrument a
new replay as described in R20_R19_TIMING_CODE_AUDIT.md.

Return all new build/identity/log/parity/benchmark/training evidence to Codex.
Codex has not sent these instructions or messages to another agent or service.

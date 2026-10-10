# R21 DSH execution and return contract

Status: INSTRUCTIONS_COMPLETE; MX capture/replay NOT_EXECUTED by Codex.
Date: 2026-10-09. No production renderer rebuild is requested in this phase.

## Preparation

Use the SAME MX Python environment, CUDA build, A100, environment flags and
dataset as the validated R20 runs. Copy this complete R21 directory into the
MX repository. Keep the R20 trainer at its existing path. `R21_SOURCE_IDENTITY`
pins the actual trainer/model/config/dataset/camera/initialization code; a
mismatch requires exporting and re-auditing the differing source, not disabling
the check. These source hashes normalize ONLY CRLF to LF for Windows/Linux
checkouts; actual byte hashes are separately recorded. Binary/checkpoint hashes
are exact bytes. Verify the delivery manifest before running.

```bash
set -euo pipefail
export R21_REPO=/home/liaoyuanjun/3dgs-renderer-benchmark
export R21_TASK="$R21_REPO/experiments/r21_densification"
export R21_STD=/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx
# Point at the EXISTING validated R20 candidate; do not prepare/build over it.
export R21_HIGS=/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_ABSGRAD_PARITY_R20_MX/gsplat-higs-r20
export R21_OUT=/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_DENSIFICATION_PARITY_R21_MX
export R21_GPU=0
test -f "$R21_HIGS/R20_BUILD_IDENTITY.json"
test ! -e "$R21_OUT"
mkdir -p "$R21_OUT/logs"
cd "$R21_TASK"
sha256sum -c R21_SHA256SUMS.txt
python R21_REGRESSION_TESTS.py > "$R21_OUT/logs/cpu_tests.log" 2>&1
python R21_CAPTURE_FIRST_DIVERGENCE.py --repo "$R21_REPO" \
  --project "$R21_STD" --backend std_ll --scene bicycle \
  --out "$R21_OUT/static_unused" --static-check > "$R21_OUT/logs/static.json"
```

`--static-check` does not import the CUDA renderer or create a run directory.
Actual capture checks both frozen main and R20 candidate identities via the
unchanged trainer and records all renderer file hashes before/after execution.
The required full sources include `gsplat/rendering.py`, `Rendering.cpp`,
`ProjectionEWA3DGSFused.cu`, `_wrapper.py`, `Utils.cuh`, `Common.h`, HiGS wrapper
and backward CUDA. Capture fails if these actual sources are unavailable.

## Capture the original trajectories

Use R20's existing HIGS/GSPLAT environment values. Do not silently change PX,
TF32 or deterministic settings; the tool records them. Run serially on the
chosen GPU to avoid competition. Output directories must be new.

```bash
for scene in bicycle room; do
  for backend in std_ll higs; do
    project="$R21_STD"
    if [ "$backend" = higs ]; then project="$R21_HIGS"; fi
    python "$R21_TASK/R21_CAPTURE_FIRST_DIVERGENCE.py" \
      --repo "$R21_REPO" --project "$project" --backend "$backend" \
      --scene "$scene" --seed 42 --iterations 30000 --stop-after 600 \
      --capture-at 1,100,599,600 --resume-every 100 --gpu "$R21_GPU" \
      --out "$R21_OUT/${scene}_${backend}" \
      > "$R21_OUT/logs/${scene}_${backend}.log" 2>&1
  done
  python "$R21_TASK/R21_COMPARE_GAUSSIAN_STATE.py" --journals \
    --left "$R21_OUT/${scene}_std_ll/steps.jsonl" \
    --right "$R21_OUT/${scene}_higs/steps.jsonl" \
    --output "$R21_OUT/${scene}_first_divergence.json"
  for step in 000001 000100 000599 000600; do
    python "$R21_TASK/R21_COMPARE_GAUSSIAN_STATE.py" \
      --left "$R21_OUT/${scene}_std_ll/iter_${step}_pre_topology.pt" \
      --right "$R21_OUT/${scene}_higs/iter_${step}_pre_topology.pt" \
      --output "$R21_OUT/${scene}_${step}_comparison.json"
  done
  python "$R21_TASK/R21_ANALYZE_THRESHOLD_CROSSING.py" \
    --left "$R21_OUT/${scene}_std_ll/iter_000600_pre_topology.pt" \
    --right "$R21_OUT/${scene}_higs/iter_000600_pre_topology.pt" \
    --output "$R21_OUT/${scene}_threshold_crossing.json"
done
```

All original training steps and scheduled evaluations through 600 are retained.
The final 30K evaluator is skipped because this diagnostic stops early. Camera
sequence generation retains the original 30K horizon. Captures occur after
backward/statistics before topology, and after Adam/zero_grad. At non-densify
steps candidate masks are previews; `decisions` only contains executed masks.

Hashing/copying all rows each step and retaining full states has substantial
host/transfer/disk overhead. It preserves math/RNG but can change scheduling.
Do not compare these timings against R18/R20 performance. Full snapshots store
raw parameters, raw gradients and Adam moments: provision several GB of CPU RAM
and disk, and use a narrow rerun window for `--capture-every-step`.

## Narrow or resume a capture

For a first difference at 127, resume BOTH original trajectories from their
own step-100 checkpoints and capture each step through 127 into new directories.
For differences before 100, start from initialization with `--stop-after` set
to the observed step. A failed run can resume from its last completed
`post_step.pt` plus verified JSON sidecar; do not append to a partial journal.

```bash
# Example narrowed std run; perform the same operation with the HiGS path.
python "$R21_TASK/R21_CAPTURE_FIRST_DIVERGENCE.py" \
  --repo "$R21_REPO" --project "$R21_STD" --backend std_ll --scene bicycle \
  --resume "$R21_OUT/bicycle_std_ll/iter_000100_post_step.pt" \
  --stop-after 127 --capture-every-step --gpu "$R21_GPU" \
  --out "$R21_OUT/bicycle_std_ll_narrow_101_127"
```

Run the journal comparison on the original complete 1–600 journals to claim
the first step since initialization. A comparison of only resumed journals
correctly reports `proves_first_since_initialization=false`. Compare the
narrowed full states by stable ID to quantify the earliest discrepancy.

## Same-state A–F and repeatability

```bash
for scene in bicycle room; do
  for base in higs std_ll; do
    python "$R21_TASK/R21_CAUSAL_REPLAY.py" \
      --repo "$R21_REPO" --reference-root "$R21_STD" --candidate-root "$R21_HIGS" \
      --state "$R21_OUT/${scene}_${base}/iter_000599_post_step.pt" \
      --history-reference "$R21_OUT/${scene}_std_ll/iter_000599_post_step.pt" \
      --gpu "$R21_GPU" --out "$R21_OUT/${scene}_causal_from_${base}_repeat1"
    python "$R21_TASK/R21_CAUSAL_REPLAY.py" \
      --repo "$R21_REPO" --reference-root "$R21_STD" --candidate-root "$R21_HIGS" \
      --state "$R21_OUT/${scene}_${base}/iter_000599_post_step.pt" \
      --history-reference "$R21_OUT/${scene}_std_ll/iter_000599_post_step.pt" \
      --gpu "$R21_GPU" --out "$R21_OUT/${scene}_causal_from_${base}_repeat2"
  done
done
```

The tool rejects A–F input mismatches in raw input, Adam/RNG, camera/GT, native
pre-intervention history, stable/native row order, counters, SH/recipe/extent
and numerical runtime settings. C–F donors must be native std_ll arm A;
historical E donors must be std_ll. Causal mode executes exactly one step.
Compare A repeat1 vs A repeat2 and B vs B with the state
comparator; retain both exact and tolerance reports. Add `--plan-only` in a new
directory to emit reviewable commands without running them. Omitting
`--history-reference` makes E a labelled same-history no-op control.

F may recover topology only by construction; inspect parameter values, raw
gradients, child random offsets and optimizer states to evaluate downstream
behavior. F is never a proposed production implementation.

## Required return package and next gate

Return these files without stripping dtype/IDs or keeping only summaries:

- Every `RUN_IDENTITY.json`, `RUN_STATUS.json`, complete `steps.jsonl`, logs,
  camera_sequence.npy and `source_evidence/` with actual frozen/candidate sources.
- Full states and `.json` hash/schema sidecars at 1,100,599,600, plus the
  narrowed first-divergence interval and causal A–F pre/post states.
- First-divergence, aligned comparisons, threshold reports, CAUSAL_PLAN/STATUS,
  and A/A, B/B repeatability results. Include R20's actual raw parity/5K reports.

Do not mutate the tested R20 candidate. Codex will evaluate H1–H6 against the
returned evidence, add a mechanism-specific failing test, and only then create
an isolated R21 production candidate if warranted. The subsequent MX gates are
R20 True AbsGrad/signed/raw-gradient parity, visibility/radii/candidate parity,
dynamic counts, 1000-step training smoke, 5K quality and renderer performance.
The final quality limits remain -.10 dB PSNR, -.002 SSIM, +.005 LPIPS.

## Machine-readable output schema

`r21.gaussian-state.v1`: `iteration`, `phase`, backend/arm/scene/seed/recipe,
camera_sequence and actual camera K/view/GT hash; stable/native IDs and next
native ID; all raw tensors/gradients; statistics; named Adam groups/states;
Python/NumPy/CPU/CUDA RNG; active SH and extent. Pre-topology adds actual
projection/canonical projection/calls, absgrad/signed2d, radii, native and
consumed visibility/history, scores/previews, executed decisions and lineage
events. Post-step captures exact resume state. `.json` sidecar records every
tensor's dtype/shape and the `.pt` SHA256. Loading uses weights_only=True.

`r21.step.v2` journals carry per-step hashes for raw-before/after, projection,
radii/visibility, raw/absolute/signed gradients, before/after statistics,
candidate masks, executed decisions, topology ID sets and RNG/optimizer input.
Gaussian row hashes are canonicalized by stable ID. `r21.comparison.v1` reports
ID set differences, metrics and mismatch IDs; `r21.threshold-analysis.v1`
reports score/scale margins, counts and algebraic A/D decomposition.

The first-topology report distinguishes lineage-event divergence from final
survivor-set divergence. A birth followed by pruning in the same transaction
can leave the same survivor set and must still be reported. Event hashes
canonicalize parent/child edges within each phase, preserving phase order.
Do not mix legacy step.v1 journals with step.v2; the comparator rejects them.

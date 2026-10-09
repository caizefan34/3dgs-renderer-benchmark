# DSH MX P4/P5 execution

Status: COMMANDS_READY; P4.1 full MX / P4.2 / P4.3 / P5 NOT_EXECUTED by Codex.
Use the environment exports and isolated preparation in `R21_BUILD_INSTRUCTIONS.md`.
Run serially on the A100. Preserve the validated R20 runtime flags and CUDA/
PyTorch environment; record GPU identity, clocks, PX, TF32 and deterministic
settings. No frozen binary, R20 candidate or original checkpoint is writable.

## P4.1: same-state opacity/visibility and numerical parity

```bash
export CUDA_VISIBLE_DEVICES=0
export R21_OLD=/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_DENSIFICATION_PARITY_R21_MX
export R21_DIAG="$R21_REPO/experiments/r21_densification"
export R20_TASK="$R21_REPO/experiments/r20_true_absgrad"
for px in 0 1 2 4; do
  for batch in 1 4; do
    dest="$R21_OUT/synthetic_px${px}_B${batch}"
    mkdir -p "$dest"
    HIGS_PX_RUNTIME="$px" "$R21_PY" "$R21_TASK/R21_VISIBILITY_REGRESSION_TESTS.py" --gpu \
      --reference-root "$R21_STD" --candidate-root "$R21_CANDIDATE" \
      --cameras "$batch" --output "$dest/results.json" > "$dest/run.log" 2>&1
  done
done
for scene in bicycle room; do
  for step in 000001 000599 000600; do
    for batch in 1 4; do
      dest="$R21_OUT/${scene}_iter${step}_B${batch}"
      mkdir -p "$dest"
      HIGS_PX_RUNTIME=2 "$R21_PY" "$R21_TASK/R21_VISIBILITY_REGRESSION_TESTS.py" --gpu \
        --reference-root "$R21_STD" --candidate-root "$R21_CANDIDATE" \
        --state "$R21_OLD/${scene}_std_ll/iter_${step}_pre_topology.pt" \
        --cameras "$batch" --output "$dest/results.json" > "$dest/run.log" 2>&1
    done
  done
done
```

Each worker consumes identical raw parameters and cotangents in separate
processes. It tests absgrad=True/False, native rendering plus dynamic topology,
opacity resets, low opacity, boundary/near-plane/large-radius cases, and actual
Bicycle/Room mismatch subsets. PX controls CUDA dispatch only in these MX
runs; CPU PX parameter plumbing is not evidence of CUDA correctness.
B4 uses deterministic camera translations, recorded in the shared fixture.
Synthetic `multi_camera` remains a two-camera stress case in the B1 suite.

Use the exact R18/R20 large-scene fixtures (not a newly initialized scene).
Their paths are deployment inputs because no fixed R20 fixture pathname is
part of the source contract. The following fails until DSH selects both
existing files; it never guesses or silently substitutes a small fixture.

```bash
: "${R21_FIXTURE_580K:?Set the validated R18/R20 approximately 580K real fixture path}"
: "${R21_FIXTURE_441M:?Set the validated R18/R20 full 4.41M real fixture path}"
"$R21_PY" - "$R21_FIXTURE_580K" "$R21_FIXTURE_441M" <<'PY'
import sys, torch, hashlib
for p,lo,hi in zip(sys.argv[1:], (570000,4400000), (590000,4420000)):
    f=torch.load(p,map_location='cpu',weights_only=True)
    assert lo <= len(f['xyz']) <= hi, (p,len(f['xyz']))
    print(p,len(f['xyz']),hashlib.sha256(open(p,'rb').read()).hexdigest())
PY
for size in 580K 441M; do
  var="R21_FIXTURE_$size"; fixture="${!var}"
  for batch in 1 4; do
    dest="$R21_OUT/${size}_B${batch}"; mkdir -p "$dest"
    HIGS_PX_RUNTIME=2 "$R21_PY" "$R21_TASK/R21_VISIBILITY_REGRESSION_TESTS.py" --gpu \
      --reference-root "$R21_STD" --candidate-root "$R21_CANDIDATE" \
      --fixture "$fixture" --cameras "$batch" --output "$dest/results.json" > "$dest/run.log" 2>&1
  done
done
```

If the large fixture files are absent, the unchanged `R20_MAKE_REAL_FIXTURE.py`
can export them from a hash-verified checkpoint plus the frozen cameras and
reference scene extent; see its `--help` and R20 build instructions. Record
the exact subset indices/protocol for 580K. Do not label a sampled scene as a
full 4.41M result. Keep the original checkpoint read-only.

Acceptance: exact visibility ID sets and radii, not counts alone; report
radii delta histogram, RGB/alpha, signed means2d, all raw parameter gradients,
native True AbsGrad, MAE/RMSE/relative error/cosine, threshold and clone/split/
prune parent candidates and Top-K. Default numerical atol=2e-6/rtol=.002 is
the existing R20 test setting; inspect near-threshold errors separately.
The saved worker tensors permit complete mismatch analysis. `PASS_SCOPED`
applies only to that fixture; it does not pass the historical/training gates.
P4 historical candidate sets use actual trainer masks, not one-step previews.

## P4.2: complete history from initialization, iterations 1–600

```bash
for scene in bicycle room; do
  for backend in std_ll higs; do
    root="$R21_STD"; if [ "$backend" = higs ]; then root="$R21_CANDIDATE"; fi
    HIGS_PX_RUNTIME=2 "$R21_PY" "$R21_TASK/R21_CAPTURE_CANDIDATE.py" \
      --repo "$R21_REPO" --project "$root" --backend "$backend" --scene "$scene" \
      --seed 42 --iterations 30000 --stop-after 600 --capture-at 1,100,599,600 \
      --resume-every 100 --gpu 0 --out "$R21_OUT/replay_${scene}_${backend}" \
      > "$R21_OUT/logs/replay_${scene}_${backend}.log" 2>&1
  done
  left="$R21_OUT/replay_${scene}_std_ll"; right="$R21_OUT/replay_${scene}_higs"
  "$R21_PY" "$R21_DIAG/R21_COMPARE_GAUSSIAN_STATE.py" --journals \
    --left "$left/steps.jsonl" --right "$right/steps.jsonl" \
    --output "$R21_OUT/${scene}_first_divergence.json"
  for step in 000001 000599 000600; do
    "$R21_PY" "$R21_DIAG/R21_COMPARE_GAUSSIAN_STATE.py" \
      --left "$left/iter_${step}_pre_topology.pt" --right "$right/iter_${step}_pre_topology.pt" \
      --atol 2e-6 --rtol .002 --output "$R21_OUT/${scene}_${step}_state.json"
  done
  "$R21_PY" "$R21_DIAG/R21_ANALYZE_THRESHOLD_CROSSING.py" \
    --left "$left/iter_000600_pre_topology.pt" --right "$right/iter_000600_pre_topology.pt" \
    --output "$R21_OUT/${scene}_threshold_crossing.json"
done
```

Check identical camera sequence, initial parameters/optimizer/RNG and scene
extent first. Compare A (xyz_gradient_accum), D (denom), max radii and all
actual clone/split/prune IDs. Journals cover every step; snapshots allow
algebraic A/D decomposition. Hash differences alone do not establish harmful
error, and `COMPARISON_COMPLETE` is not PASS. If a first crossing lies between
snapshots, use `--resume ...post_step.pt --capture-every-step` on both original
trajectories to narrow it. Do not transplant reference history/masks/radii.

At iteration 600 demand recovery of **Bicycle 331 / Room 68 splits** and
high agreement of the actual selected stable-ID sets. Counts matching is
insufficient. Important threshold-crossing mismatches must be zero or have a
specific audited numerical explanation. If 331/68 is not recovered, keep
the root-cause-complete verdict FAIL/UNRESOLVED.

## P4.3: paired 1000-step smoke

Continue each verified replay from its own step-600 post-step state with
the same 30K camera-sequence horizon. These complete checkpoint restores
retain raw parameters, Adam, IDs, A/D history and RNG.

```bash
for scene in bicycle room; do
  for backend in std_ll higs; do
    root="$R21_STD"; if [ "$backend" = higs ]; then root="$R21_CANDIDATE"; fi
    HIGS_PX_RUNTIME=2 "$R21_PY" "$R21_TASK/R21_CAPTURE_CANDIDATE.py" \
      --repo "$R21_REPO" --project "$root" --backend "$backend" --scene "$scene" \
      --seed 42 --iterations 30000 --stop-after 1000 --capture-at 700,800,900,1000 \
      --resume "$R21_OUT/replay_${scene}_${backend}/iter_000600_post_step.pt" \
      --resume-every 100 --gpu 0 --out "$R21_OUT/smoke_${scene}_${backend}" \
      > "$R21_OUT/logs/smoke_${scene}_${backend}.log" 2>&1
  done
done
```

Inspect finite per-step losses, raw parameters/gradients/Adam moments, expected
optimizer tensor row counts, finite nonnegative True AbsGrad, topology events
and dynamically updated Gaussian counts. Capture completion alone is not a
numerical health verdict. Opacity reset is also directly tested by P4.1 since
the scheduled reset may occur after this smoke window.

## P5: paired 5K, only after DSH signs off P4

DSH writes `$R21_OUT/P4_ACCEPTANCE.json` after reviewing actual results with
keys `P4.1`, `P4.2`, `P4.3`, each containing `status: PASS`, evidence paths and
an explanation for every tolerated mismatch. No such PASS file is shipped.

```bash
"$R21_PY" - "$R21_OUT/P4_ACCEPTANCE.json" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]))
assert all(a[k]['status']=='PASS' and a[k]['evidence'] for k in ('P4.1','P4.2','P4.3'))
PY
for scene in bicycle room; do
  for backend in std_ll higs; do
    root="$R21_STD"; if [ "$backend" = higs ]; then root="$R21_CANDIDATE"; fi
    test ! -e "$R21_OUT/p5_${scene}_${backend}"
    HIGS_PROJECT="$root" HIGS_PX_RUNTIME=2 "$R21_PY" "$R20_TASK/R20_TRAINER.py" \
      --backend "$backend" --scene "$scene" --seed 42 --iterations 5000 \
      --gpu 0 --densify-mode absgrad --outdir "$R21_OUT/p5_${scene}_${backend}" \
      > "$R21_OUT/logs/p5_${scene}_${backend}.log" 2>&1
  done
done
"$R21_PY" - "$R21_OUT" <<'PY'
import json,sys,pathlib,math
root=pathlib.Path(sys.argv[1]); verdict={}; passed=True
for scene in ('bicycle','room'):
    a=json.load(open(root/f'p5_{scene}_std_ll/train_results.json'))
    b=json.load(open(root/f'p5_{scene}_higs/train_results.json'))
    d={k:b['final_eval'][k]-a['final_eval'][k] for k in ('psnr','ssim','lpips')}
    ok=all(math.isfinite(x) for x in d.values()) and d['psnr']>=-.10 and d['ssim']>=-.002 and d['lpips']<=.005
    verdict[scene]={'status':'PASS' if ok else 'FAIL','delta':d,'std_N':a['final_N'],'higs_N':b['final_N']}
    passed &= ok
(root/'P5_QUALITY.json').write_text(json.dumps(verdict,indent=2)+'\n')
print(json.dumps(verdict,indent=2)); sys.exit(0 if passed else 1)
PY
```

Missing LPIPS is a failed/incomplete quality gate, not zero. Retain per_iter,
densification events, counts, metric series and raw logs. A quality FAIL ends
this schedule; **do not run 30K**. Passing permits a later 30K/Time-to-Quality
decision; these commands do not launch it.

## Fresh renderer timing

After P4, verify candidate identity again and run the unchanged R20
`R20_ABSGRAD_MICROBENCH.py` separately on std and candidate, for both large
fixtures and absgrad modes, at B1/B4. Use the `fixtures.pt` emitted by the
matching P4.1 run and export its `fixture_*` entry to a new `.pt` file, so
camera state is exactly the tested state. Example after setting
`R21_BENCH_FIXTURE` to that exported single-fixture path:

```bash
: "${R21_BENCH_FIXTURE:?Set the P4-verified single-fixture B1 or B4 path}"
"$R21_PY" "$R21_TASK/R21_PREPARE_CANDIDATE.py" --verify --candidate-root "$R21_CANDIDATE"
for backend in std higs; do
  root="$R21_STD"; if [ "$backend" = higs ]; then root="$R21_CANDIDATE"; fi
  HIGS_PX_RUNTIME=2 "$R21_PY" "$R20_TASK/R20_ABSGRAD_MICROBENCH.py" \
    --root "$root" --backend "$backend" --absgrad --real-fixture "$R21_BENCH_FIXTURE" \
    --warmup 20 --iterations 200 --output "$R21_OUT/bench_${backend}.json"
done
```

Repeat with unique output names for each size/batch and without `--absgrad`
for signed mode. Preserve raw forward/backward event CSV, mean/median/p99,
wall time and peak allocated/reserved VRAM. Do not time instrumented historical
capture as renderer performance. R20's 6.3% advantage is not a P4 measurement.

# SelectiveAdam source audit — final sprint

Status: **STOP BEFORE BENCHMARK** (2026-09-18).

## Scope and frozen baseline

The requested baseline is `baseline/reference_v1`, with no HiGS, Candidate C,
C42, sparse gradients, or other training changes. The relevant Reference V1
path is:

1. `baseline/reference_v1/trainer.py:317` renders one selected camera.
2. `baseline/reference_v1/trainer.py:335-345` derives
   `visibility_filter = (meta["radii"][0] > 0).any(dim=-1)`. It is used only
   for densification statistics and screen-radius tracking.
3. `baseline/reference_v1/gaussian_model.py:203-227` creates one persistent
   standard `torch.optim.Adam` with five parameter groups: xyz, SH, opacity,
   scaling, and rotation.
4. `baseline/reference_v1/trainer.py:427-429` calls
   `model.optimizer.step()` and `zero_grad`; there is no optimizer flag,
   SelectiveAdam entry point, fused-Adam selection, mask transfer, or masked
   state update in this pipeline.

Reference V1 constructs `torch.optim.Adam(...)` without `fused=True`; no
Reference V1 fused masked CUDA kernel is present. The HiGS implementation,
by contrast, JIT-compiles a custom CUDA extension at first use.

Thus Reference V1 does expose a current-camera visibility predicate, but it
does not feed it to the optimizer.

## Only discovered masked-Adam implementation

The sole executable implementation is explicitly HiGS-specific:

- `benchmark/higs_masked_adam.py:43-92` JIT-compiles the CUDA kernel
  `masked_adam_kernel` with `torch.utils.cpp_extension.load_inline`
  (`benchmark/higs_masked_adam.py:132-139`).
- `benchmark/higs_masked_adam.py:152-195` applies it to every optimizer
  parameter group. Each tensor is reshaped to `[N, D]`; true mask rows update
  parameter, `exp_avg`, and `exp_avg_sq`.
- `benchmark/run_higs_train_benchmark.py:808-837` fetches the mask from the
  HiGS renderer handle's patched `_cull_cache["train"]`. It calls this a
  **union-visibility** mask for the latest train forward and rejects a
  non-HiGS renderer handle or a missing HiGS cull cache.
- `benchmark/run_higs_train_benchmark.py:1201-1207` selects
  `masked_adam_step` only for `--masked-adam`; otherwise it calls `opt.step()`.

This mask is not a Reference V1 optimizer input and is not proven to be the
single current-camera `radii > 0` set required here. It is a HiGS
train-forward union cache, so using it would mix the forbidden HiGS path.

## Semantic finding

The HiGS source itself states the distinction at
`benchmark/higs_masked_adam.py:13-17`: false-mask rows are *frozen completely*
(parameters and moments unchanged), whereas stock Adam continues decaying
moments on zero-gradient rows. The kernel branch at line 56 returns before
writing `p`, `m`, or `v`. `scripts/higs/probe_masked_adam.py:100-132` checks
that false rows are bit-identically frozen; it does not establish equivalence
to standard Adam for those rows.

Therefore a partial visibility mask is intentionally a different optimizer
trajectory. All-true masks are approximately fused-Adam-equivalent; partial
masks are not standard-Adam-equivalent by design.

## Decision

No compatible SelectiveAdam path exists in Reference V1. Porting the HiGS
kernel/cull cache or adding a new Reference V1 masked optimizer would be new
implementation work, change the specified optimizer semantics for invisible
rows, and violate the no-HiGS/no-algorithm-change constraint. Per the
semantic gate, the ablation stops before the microbenchmark.

## Audit commands

```powershell
rg -n -i "selective.?adam|visible.?adam|visibility.?mask|adam.*mask" baseline src scripts experiments reports configs variants
rg -n "Adam|optimizer|step\\(|radii|visible|isect|meta" baseline/reference_v1/trainer.py baseline/reference_v1/gaussian_model.py
rg -n "masked_adam_step|higs_masked_adam|masked-adam" .
```

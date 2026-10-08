# FastGS real-camera shadow gate: correctness passed, training claim open

2026-09-27. This is an isolated diagnostic of one split family, not a faster
trainer or a measured reconstruction gain. Every image in this check is at the
author's fixed 1600×1066 effective resolution; no low-resolution optimization
is involved.

## Reproduction boundary

- Author FastGS commit: `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`.
- EPIC-05 A100 GPU 2, isolated author checkout
  `/root/codex_fastgs_audit_20260927`; no author source was changed.
- Camera: room training image `DSCF4668`; 4099-step checkpoint and pre-event
  4100-step PLY contain 651,547 rows and match by `torch.equal` for all six
  parameter tensors. Raw result: `fastgs_checkpoint_ply_identity_result.json`.
- PLY SHA256 `64160970e5ea580aea4f79278a8fbe334d8d81ef6e304a903626843cc48f85bc`;
  native 4100 split-selection mask SHA256
  `d2a393e0513f71bbd19996a9c5044da9dde7faf11740939f845b35c9b7c8b0a7`.
  The mask selects 9,345 split parents. Parent `415040` is among them.
- Local/remote probe SHA256 `a27ba2fd80924d9c85942d9c348866af73828c073f3cf978a6a21a586266bfd7`;
  CUDA source SHA256 `23cc0f80a4e7c9cb28294504d27660370a72d13c4ef6c3c72a07496057681b2b`;
  extension loader SHA256 `82cb90f06b069329936841b167d682a9c3fd924b5ff1f91216ee0cda2bb4a04b`.
  Sources are `fastgs_real_shadow_cuda_probe.py`,
  `fastgs_shadow_gate_cuda.cu`, and `fastgs_shadow_gate_renderer.py`.

## Result

The probe used author CUDA projected rows and depth-sorted tile lists. For a
single selected native-rule split family, it retained parent plus children as
fixed candidate support, applied the optical-depth gate, and compared the
`g=0` and `g=1` full images against separate native pre/post renders. Maximum
absolute RGB error at both endpoints was `1.7881393e-7`; zero pixels exceeded
the prespecified `2e-5` threshold. Three selected pixels' GPU gate VJPs
matched independent FP64 fixed-support references within `6.09e-15` absolute.
The probe did not check all pixels' VJPs against independent autograd.

The separate full-image diagnostic gate kernel took median `1.9472 ms` at
`g=0` and `1.9464 ms` at `g=1` (six CUDA-event repetitions each). These timings
exclude candidate construction, projection, sorting, all-family gate gradients,
ordinary backward, optimizer, rollback, data loading, and saving. The single
family's tile list has 866,239 instances versus 866,228 before split and
866,229 after split. `fastgs_shadow_cuda_v3_result.json` and the preserved
`fastgs_shadow_cuda_v3_20260927.log` contain the raw run.

For the **entire** 9,345-parent native split event on this same camera, a
separate static count found 866,228 instances before split, 872,713 after,
and 893,886 if both parents and children are retained: +21,173 (2.43%)
relative to native post-split. Native pre/post forward + original loss backward
medians were 6.158/6.217 ms. These are endpoint costs only, not an actual
all-family gate implementation or saved training work. See
`fastgs_event_split_cost_result.json`.

The earlier `__expf` diagnostic source is preserved as
`fastgs_shadow_gate_cuda_v2_failed.cu`; its prior unsaved run had one endpoint
pixel outside `2e-5`. The current source differs only by replacing `__expf`
with `expf`, corresponding to the author path. The v2 raw output was not
preserved, so the v2 numerical observation is not promoted to a reproducible
result here.

## Gate before training

Implement per-family VJPs for a whole selected event and measure union
construction, projection/sort, forward/backward and rollback. A separate
full-image pass near 1.95 ms is already substantial relative to a ~6.2 ms
native forward/loss/backward step, so a probation policy must demonstrate
amortized net savings rather than assume the 2.43% tile count is cheap. Only
then test matched-state full-resolution training, long-horizon usefulness,
untouched views/scenes, full wall time, and prior-art novelty.

# SAD-GS hybrid SH last-mask CPU gate result

**Outcome: the stated failure mode is confirmed as a source-level conditional behavior and reproduced by the CPU model.** With `batch_size=2`, a Gaussian that has nonzero accumulated SH gradient from camera 1 but is invisible in camera 2 is omitted by the last-camera mask. Its parameter and Adam moments remain unchanged in the sparse SH update; the subsequent `shoptimizer.zero_grad(set_to_none=True)` clears the accumulated gradient. A Gaussian visible in camera 2 is updated, which serves as a positive control. Under a union mask, the camera-1-only Gaussian would update.

The frozen source is the clean author checkout at `029814fb5ed773e4a794cec5115951a64f610ab8`. The code path and exact source hashes are in `FROZEN_CONDITIONS.md`. The runnable standard-library CPU model is `last_mask_cpu_gate.py`, and its saved output is `cpu_gate_result.json`.

The script's assertions all pass:

- camera-1-only Gaussian has accumulated SH gradient `[1.25, -0.5]`;
- final camera mask is false for that Gaussian, so both SH parameters and moments stay unchanged;
- the gradient is then modeled as `None` after `set_to_none=True`;
- the last-camera-visible control updates; and
- a union visibility mask would update the first Gaussian.

This establishes the control-flow implication under the stated condition, not how often the condition occurs in real batch training. It is pure-Python float arithmetic, not PyTorch autograd, IEEE FP32 emulation, the CUDA optimizer, or a training run. It does not establish actual scene incidence, measured quality impact, or aggregate training behavior. No author source was changed, no GPU was used, and no training was run.

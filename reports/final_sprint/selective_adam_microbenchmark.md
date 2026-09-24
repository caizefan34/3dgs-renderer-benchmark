# SelectiveAdam static optimizer microbenchmark

Status: **NOT RUN — early semantic/integration stop**.

The required room, bicycle, and garden measurements were not started. The
only available masked implementation requires the HiGS renderer's
`_cull_cache["train"]` union-visibility cache and its JIT CUDA kernel; the
Reference V1 standard-Adam training loop has neither. Constructing an adapter
or deriving a new Reference V1 masked path would be an implementation change
outside this standalone ablation.

No N_total, N_visible, visible fraction, optimizer time, iteration time,
kernel count, or memory-traffic figures are reported because none were
measured under the required matched pipeline. Historical HiGS timings are
not substituted: they include a different renderer/culling setup and would
not answer this research question.

Early gate result: **STOP**. There is no valid three-scene E2E signal on
which to apply KEEP/DROP performance thresholds.

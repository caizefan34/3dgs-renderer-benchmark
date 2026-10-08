# Fixed-batch covariance/opacity pullback cost gate

2026-09-27. A bounded diagnostic following the native stage profile.

In pinned FastGS, all model parameters stay fixed within late 32/64-view
accumulation blocks. Covariance and activated opacity are view independent.
For a fixed block, the chain rule permits summing their view gradients before
applying the shared parameter Jacobian once. This uses current parameters;
it does not reuse stale view gradients or omit zero-gradient Adam drift.
The chain rule is elementary and is not an originality claim.

Test native rendering against the author's existing `cov3D_precomp` interface,
with covariance/opacity built once per block and their pullback deferred until
after all views. No new CUDA. Include construction, dense buffer accumulation,
and the final pullback in timing. Use 64 distinct original training cameras
drawn by PCG64 seed 20260927, original effective width 1600 and original loss.
First use room@30K (existing pinned PLY), then garden@4100 if room passes.
The earlier garden state is only a cost stress case, not late-stage training.

Before timing, sum gradients over the first four selected views for two native
replays and the deferred implementation. Require maximum rendered RGB error
<=1e-4 and every raw parameter gradient to satisfy relative L2 <=1e-3 OR
maximum absolute error <=1e-7. Record native replay discrepancies too. Do not
relax these gates after seeing results. Failure closes this implementation.

If parity passes, warm both 64-view blocks once and measure five pairs in
alternating order. No parameter update occurs; this is a frozen-state ordinary
forward/loss/backward block benchmark, not full training. Require median wall
ratio <=0.85 and all five ratios <1 before considering trainer integration.
The optimizer, topology, data loading, saving and initialization are excluded;
they still have to be counted in complete-training claims. Repeated blocks do
not constitute independent training seeds. A pass preserves native quality
semantics only; it provides no higher-quality mechanism or original algorithm.

If this cost gate fails, stop before garden or a trainer/CUDA implementation.
Do not use it to reopen visibility-lazy Adam or the failed old-gradient cache.

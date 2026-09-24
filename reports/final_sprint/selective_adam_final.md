# SelectiveAdam final decision

```text
SELECTIVE_ADAM_STATUS: COMPLETE

SOURCE_PATH:
Reference V1: baseline/reference_v1/gaussian_model.py:203-227 creates
torch.optim.Adam; baseline/reference_v1/trainer.py:427-429 calls step().
The only masked implementation is benchmark/higs_masked_adam.py, invoked by
benchmark/run_higs_train_benchmark.py with a HiGS _cull_cache["train"]
union-visibility mask.

CORRECTNESS: FAIL (the requested Reference V1 paired semantic gate is not
implementable from an existing path; partial-mask semantics are intentionally
different from standard Adam)

3SCENE_OPTIMIZER_SPEEDUP: NOT_MEASURED

3SCENE_E2E_SPEEDUP: NOT_MEASURED

5K_TRAINING: NOT_RUN (early stop)

QUALITY: NOT_MEASURED

13SCENE_STATUS: NOT_RUN (early stop)

13SCENE_GEOMEAN_OPTIMIZER_SPEEDUP: NOT_MEASURED

13SCENE_GEOMEAN_ITERATION_SPEEDUP: NOT_MEASURED

VISIBLE_FRACTION_CORRELATION: NOT_MEASURED

FINAL_VERDICT: SELECTIVE_ADAM_NEUTRAL
```

The neutral verdict means **unsupported as a standalone Reference V1
ablation**, not measured neutral performance. It avoids both an invalid HiGS
composition and a newly implemented optimizer path. The limiting factor is
integration/semantic incompatibility: false-mask rows freeze parameter and
Adam state, while standard Adam advances the state even with zero gradient.

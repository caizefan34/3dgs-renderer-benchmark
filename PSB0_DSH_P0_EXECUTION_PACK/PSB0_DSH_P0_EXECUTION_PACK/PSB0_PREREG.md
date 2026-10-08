# PSB0-P0 Preregistration

## Gate 0 — remote
EPIC-05 only.

## Gate 1 — exact algebra
Derive a segment summary S(a:b) with a binary composition operator ⊗.

Need explicit proof of:
- closure;
- associativity (or a scan-compatible associative lifted representation);
- ability to recover every backward quantity currently obtained by bucket replay.

If no such representation exists:
`DROP_NO_ASSOCIATIVE_FORM`.

## Gate 2 — prior art
Fresh search.
Direct collision => `DROP_PRIOR_ART_COLLISION`.

## Gate 3 — state size
The summary must be no larger than the current checkpoint state per native
segment after alignment/metadata.

If larger:
`DROP_STATE_EXPANSION`.

## Gate 4 — operation model
Using real bucket-length traces from room/garden/bicycle, compare:
- current replay FLOPs;
- scan transform FLOPs;
- loads/stores;
- synchronization depth;
- shared memory.

PASS only if the theoretical model predicts >=20% reduction in
state-recovery work in >=2/3 scenes and no >5% increase in remaining scene.

Failure:
`DROP_OPERATION_COUNT`.

## Gate 5 — exact CPU/oracle replay
Implement scalar/vector reference only.

For frozen traces:
- recovered T/state equality must be bit-exact where mathematically expected;
- otherwise rel error <=1e-7;
- reconstructed parameter gradients cosine >=0.9999999 and relL2 <=1e-6.

Failure:
`DROP_NUMERICAL_MISMATCH`.

## Gate 6 — strong-backend economics
Map recoverable work to measured strong-backend timing.

Require conservative optimistic E2E >=1.05x in >=2/3 scenes.
Remaining scene >=1.02x.

Failure:
`DROP_UNECONOMIC`.

## Verdicts
- KEEP_FOR_PSB1_MINIMAL_CUDA
- DROP_NO_ASSOCIATIVE_FORM
- DROP_PRIOR_ART_COLLISION
- DROP_STATE_EXPANSION
- DROP_OPERATION_COUNT
- DROP_NUMERICAL_MISMATCH
- DROP_UNECONOMIC
- BLOCKED_SOURCE
- BLOCKED_DATA

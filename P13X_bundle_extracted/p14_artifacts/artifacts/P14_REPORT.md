# P14 Report - FreezeGS / Optimization-Induced Stationarity Oracle

Verdict: **DROP** (O2_HARD_DROP)

## 1. OBSERVED

P13-X showed natural stationarity does not survive dense Adam: dirty-intersection fraction = 1.0 at every epsilon (X3), no K<=64 DCT basis reaches cosine 0.99 (X5), and the measured E2E is dominated by backward (57% room / 47% bicycle) and optimizer (17% room / 33% bicycle).


## 2. EVIDENCE

- O1 Stage A q=0.1: median R_train = 0.8868152715784845, viable cells 6/6, cells audit>1%: 0

- O1 Stage A q=0.2: median R_train = 0.9803643460457225, viable cells 6/6, cells audit>1%: 0

- O1 Stage A q=0.25: median R_train = 0.9935575004665768, viable cells 6/6, cells audit>1%: 0

- O1 Stage A q=0.3: median R_train = 0.9989847183149625, viable cells 6/6, cells audit>1%: 0

- O1 Stage A q=0.5: median R_train = 0.9999960764409508, viable cells 6/6, cells audit>1%: 0

- O1 per-H pass checks (Stage A + Stage B): q=0.1: H32=PASS,H8=PASS,H128=FAIL -> O1_PASS=False; q=0.2: H32=PASS,H8=PASS,H128=PASS -> O1_PASS=True; q=0.25: H32=PASS,H8=PASS,H128=PASS -> O1_PASS=True

- O2 q=0.2 H32: median R_stream(max) = 0.9065, median R_rebuild = 0.9939, max FP32 cache = 24.35 GB, cells cache>24GB = 1, PASS = False

- O2 q=0.2 H128: median R_stream(max) = 0.9065, median R_rebuild = 0.9983, max FP32 cache = 24.35 GB, cells cache>24GB = 1, PASS = False

- O2 q=0.25 H32: median R_stream(max) = 0.9475, median R_rebuild = 0.9941, max FP32 cache = 19.12 GB, cells cache>24GB = 0, PASS = False

- O2 q=0.25 H128: median R_stream(max) = 0.9475, median R_rebuild = 0.9983, max FP32 cache = 19.12 GB, cells cache>24GB = 0, PASS = False


## 3. HYPOTHESIS

Optimization-induced stationarity (freeze G_F params + Adam state within a micro-epoch of H steps while G_F still fully participates in I = R(G_A, G_F)) yields an exact compiled frozen context that can compress forward, backward, and optimizer work while preserving training progress on the cluster cameras and audit generalization.


## 4. TEST

Preregistered gates O1 (optimization viability, R_train/P_audit/freeze-bitwise), O2 (compiled stream structure, R_stream/R_rebuild/cache), O3 (conservative E2E oracle with fixed floors, charged cache build and invalidation, real masked-optimizer benchmark). Frozen q grid {0.10,0.20,0.25,0.30,0.50}, H grid {8,32,128}, 6 scene-cluster cells, clusters fixed before any O1 result.

Integrity note: the first Stage A/B pass contained an inverted stream-labeling bug (active-run starts recorded as frozen segments; frozen-isect count recorded as active), detected via a mathematical impossibility (R_stream must be non-decreasing in q because top-K active sets are nested, but measured values decreased). After fixing tile_stream_hashes and adding a synthetic-stream selftest, ALL Stage A and Stage B units were regenerated in a single clean re-run; the final artifact set contains only fixed-code units.


## 5. CONCLUSION

**DROP** - O2_HARD_DROP.

Observed mechanism (both measured on the fixed-code units): (1) the active set is chosen by gradient energy, and high-energy Gaussians are precisely the large-footprint ones - across the 6 cells the active fraction of intersections spans 0.43-1.00, so freezing the low-energy majority leaves the compiled stream at R_stream 0.74-1.00 (median 0.91 at q=0.20, 0.95 at q=0.25, H=32), far above the 0.30 gate and the 0.50 hard-drop line; (2) active motion invalidates nearly every tile's frozen-segment decomposition within one micro-epoch (median R_rebuild 0.994 at H=32 and 0.998 at H=128, vs gate 0.15 and hard line 0.40). Optimization-induced stationarity preserves training progress (O1) but does NOT make the rendered streams compressible or the frozen context cache-stable under this selector definition.


## Prior-art boundary

See P14_PRIOR_ART_BOUNDARY.md. Novelty status: UNPROVEN; no novelty or speedup claim is made. The masked-optimizer benchmark (SelectiveAdam fallback: fused kernel absent from the frozen build; semantically-equivalent masked torch Adam used) is prior art, not a P14 contribution.

# CGT-TI0 — Training Integration Oracle: REPORT

- stage: TI0-A (UPDATE-COST ORACLE) complete; lane TERMINAL_AT_TI0_A
- ts: 2026-10-04 20:06:15
- scene: room | ckpt: iter_15000.pt | N=901818 | views=[15, 125, 129, 133, 193, 254] | rank=4 | fit_iters=200
- gpu_provenance: physical_gpu=0 mem_used_mib=1 util_pct=0 (clean<500MiB required, S19)

## OBSERVED
- TI0-A per-view (footprint-stable, theta_t footprints reused at theta_t1):
  - view 15: footprints=10 surv=10 match=10 dirty=6 (60.0%) wdirty=36.6% shift_med=1.18e-02 Tcan=0.311ms Tcgt=0.015ms refit_all=3568.2ms cert_dirty=666.0ms
  - view 125: footprints=10 surv=10 match=10 dirty=2 (20.0%) wdirty=16.0% shift_med=2.23e-04 Tcan=0.343ms Tcgt=0.014ms refit_all=829.8ms cert_dirty=496.0ms
  - view 129: footprints=10 surv=10 match=10 dirty=2 (20.0%) wdirty=21.7% shift_med=0.00e+00 Tcan=0.158ms Tcgt=0.015ms refit_all=693.6ms cert_dirty=484.0ms
  - view 133: footprints=10 surv=10 match=10 dirty=7 (70.0%) wdirty=64.4% shift_med=8.05e-03 Tcan=0.253ms Tcgt=0.012ms refit_all=3957.1ms cert_dirty=1354.5ms
  - view 193: footprints=10 surv=10 match=10 dirty=2 (20.0%) wdirty=15.9% shift_med=4.99e-03 Tcan=0.121ms Tcgt=0.017ms refit_all=1400.7ms cert_dirty=191.6ms
  - view 254: footprints=10 surv=10 match=10 dirty=5 (50.0%) wdirty=40.4% shift_med=7.20e-03 Tcan=0.403ms Tcgt=0.017ms refit_all=2517.7ms cert_dirty=698.3ms
- aggregate: dirty_fraction=0.35 weighted_dirty=0.29151009273021 teacher_shift_median=0.006091690978623804 teacher_shift_max=0.009197480029438654
- aggregate timing: T_canonical=0.2821119874715805 ms T_cgt=0.015359999611973763 ms T_update_A=1959.1939449310303 T_update_B=1692.1075582504272 T_update_C=581.0040235519409
- ratios: R_update_A=6944.738373190881 R_update_B=5997.999494512396 R_update_C=2059.4800978121157 | S_effective_A=0.000143992776454054 S_effective_B=0.00016672074140059237 S_effective_C=0.00048554660215832397

## EVIDENCE
- TI0-A gate (S5): S_effective_B=0.00016672074140059237 -> DROP_UPDATE_COST (thresholds: PASS>=1.15, MARGINAL>=1.0, DROP<=1.0)
- dirty-work oracle (S6): OLD coeffs (route-A L-BFGS-B fit at theta_t) re-certified against NEW teacher at theta_t1 after one real canonical optimizer step; failed=dirty.
- T_update_A=full per-tile refit; T_update_B=certify-all + refit-dirty-only (realistic); T_update_C=certify-all + reuse-passing.
- T_canonical/T_CGT = F0 fused vs teacher kernel on SAME per-tile scope (inherits frozen F0; matches upstream ~3x).

## INFERENCE
- Update/rebuild cost erases the F0 forward gain (S_effective_B<=1.0) -> DROP_UPDATE_COST, close integration lane (S5).
- weighted_dirty indicates whether incremental update (strategy B/C) can help: high weighted dirty => refit dominates.

## HYPOTHESIS
- If weighted_dirty is low and teacher_shift < cert gate (0.02), CGT state is largely reusable across steps -> incremental update viable.
- If weighted_dirty ~1.0 every step, incremental update unlikely to help; full refit required.

## NOT_MEASURED
- TI0-B (forward+backward composition), TI0-C (short time-to-quality): NOT_REACHED
- Peak VRAM / DRAM traffic / L2 / occupancy / launch count: NOT_MEASURED (S20; requires nsys/pynvml profiling run).
- Projected end-to-end training speedup (Amdahl): NOT_COMPUTED until TI0-B measured renderer/training fraction.

## CONCLUSION
- TI0-A verdict: DROP_UPDATE_COST (S_effective_B=0.00016672074140059237). Final verdict: DROP_UPDATE_COST
- F0 forward speed is NOT substituted for training speed; update + backward + recompute costs accounted separately (S24).
- Downstream artifacts are NOT_REACHED until their stages execute on a clean GPU.

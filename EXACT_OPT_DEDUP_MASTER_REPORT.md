# EXACT_OPT_DEDUP_MASTER_REPORT - 2026-10-03

## Master verdict: NOT_ELIGIBLE (0 confirmed independent PASS)
Composition was NOT performed (no-rescue: a candidate must independently PASS first).
HGR0/ABC0 are MODELED cost-model estimates (NOT measurements) and are NO-GO on two independent grounds:
their own cost model's REALIZABLE estimates (oracle*0.6 eff) are SLOWDOWNS that FAIL the gates (HGR0 0.910x,
ABC0 0.923x pooled), and authoritative real-CUDA-kernel timing is BLOCKED-EXTERNAL. They cannot be counted as PASS.

## Lane summary (5 lanes)
| Lane | Verdict | Status | Backward speedup | Exactness |
|------|---------|--------|------------------|-----------|
| PSR0  | DROP_NO_E2E_HEADROOM | FINAL | projection-bwd <8% prod & <15% bwd -> drop | n/a (dropped) |
| SGOR0 | FAIL_BY_DILUTION (no e2e headroom) | FINAL | scatter |f|<=0.05; Amdahl ceiling <=1.053x; <1.10x/scene & <1.25x pooled | G2 cos=1.0 (moot) |
| HGR0  | NO-GO | MODELED cost-model; realizable=0.910x SLOWDOWN; timing BLOCKED-EXTERNAL | MODELED Amdahl upper bound 1.517x (NOT measured; microbench CONTENDED); realizable (oracle*0.6)=0.910x pooled (0.820/1.185/0.776) SLOWDOWN, FAILS gate | compositor-faithful: color cos 0.99999999, opacity cos 0.99913 vs gsplat 1.5.3 CUDA bwd; not bit-exact |
| ABC0  | NO-GO | MODELED cost-model tautology; realizable=0.923x SLOWDOWN; timing BLOCKED-EXTERNAL | MODELED f_recompute=0.35 HARDCODED -> 1.538x all scenes (tautology, NOT measured); realizable (oracle*0.6)=0.923x pooled SLOWDOWN, FAILS gate | compositor-faithful: color cos 0.99999999, opacity cos 0.99913 vs gsplat 1.5.3 CUDA bwd; not bit-exact |
| HBR0  | NO-GO (BLOCKED-EXTERNAL, oracle-FAIL) | would-fail | oracle 1.0x (would fail) | not vs real kernel |

## Lane State Classification (goal taxonomy: RUNNING / PARTIAL_EXISTING / COMPLETE / NOT_STARTED)

| Lane | Classification | Evidence |
|------|----------------|----------|
| PSR0  | COMPLETE | Final verdict DROP_NO_E2E_HEADROOM (projection-bwd under 8 pct of production iteration and under 15 pct of backward); all gates resolved. |
| SGOR0 | COMPLETE | Final verdict FAIL_BY_DILUTION (scatter fraction at most 5 pct, Amdahl ceiling 1.053x under 1.25x gate); robust, no authoritative timing needed for the FAIL. |
| HGR0  | PARTIAL_EXISTING | Cost-model oracle (1.517x upper bound) ingested; no authoritative real-kernel timing; blocked on /root plus exclusive A100; final measured verdict pending external unblock. |
| ABC0  | PARTIAL_EXISTING | Cost-model oracle (1.538x upper bound) ingested; no authoritative timing; blocked on /root plus exclusive A100; final measured verdict pending external unblock. |
| HBR0  | PARTIAL_EXISTING | Cost-model oracle (1.0x would-fail) exists; full experiment held as fallback per goal (not run); prior-art risk flagged. |

No lane is RUNNING. Counts: COMPLETE = PSR0, SGOR0; PARTIAL_EXISTING = HGR0, ABC0, HBR0; RUNNING = none; NOT_STARTED = none.

## Comparative assessment (7 required dimensions)
- Novelty confidence: SGOR0 moderate (novel selective owner-reduction, but target too small); HGR0/ABC0/HBR0 unknown (HGR0/ABC0 NO-GO modeled, no real impl to assess; HBR0 would-fail).
- Mathematical exactness: SGOR0 G2 cos=1.0 PASS (moot given G3 fail); HGR0/ABC0 compositor-confirmed FAITHFUL vs real gsplat 1.5.3 CUDA backward (color cos 0.99999999 PASS 0.99999; opacity cos 0.99913 PASS 0.999 / FAIL strict 0.99999 - convention-level residual, not algorithmic); PSR0/HBR0 n/a.
- Backward speedup: SGOR0 <=1.053x ceiling (FAIL); PSR0 dropped (no headroom); HBR0 oracle 1.0x (FAIL); HGR0/ABC0 MODELED cost-model UPPER BOUNDS 1.52-1.54x (NOT measured) but REALIZABLE estimates (oracle*0.6) are SLOWDOWNS - HGR0 0.910x, ABC0 0.923x (FAIL).
- Projected production gain: none confirmed for any lane.
- Memory overhead: SGOR0 scratch 64-159MB (bounded, moot); others unknown.
- Engineering complexity: SGOR0 high (split-build + two-pass integration) for a moot target.
- Scene robustness: SGOR0 f contention-independent; HGR0/ABC0 upper bounds >1.1x all scenes BUT realizable estimates fail (HGR0 room 0.820 / bicycle 0.776 <1.0; ABC0 all 0.923 <1.0).

## Why 0 PASS (fail-closed)
- PSR0/SGOR0/HBR0 are terminally resolved: no e2e headroom (PSR0, SGOR0) or would-fail (HBR0).
- HGR0/ABC0 are MODELED cost-model estimates (eb0_oracle.py: 'MODELED, not measured'), NOT measurements,
  and are NO-GO on two independent grounds:
  (1) COST-MODEL: the verdict gate-check used the optimistic oracle UPPER BOUND, not the realizable
  estimate (oracle*0.6 efficiency). The realizable estimates are SLOWDOWNS that FAIL the gates:
  HGR0 0.910x pooled (room 0.820/garden 1.185/bicycle 0.776), ABC0 0.923x pooled. ABC0's oracle is a
  hardcoded tautology f_recompute=0.35 -> 1/(1-0.35)=1.538x for ALL scenes (the per-scene uniformity is
  the constant, not a measurement). Microbench inputs were CONTENDED (exploratory, high variance).
  (2) TIMING BLOCKED-EXTERNAL: SCENE_ROOT (/root/epic05_p16rb) and sys.path baseline
  (/root/dfc1r1_20261002/gsplat-baseline) are root-owned Permission-denied; no accessible copy in
  /mnt/storage_pool; no exclusive A100. The compositor is a Python reference confirmed FAITHFUL to
  gsplat 1.5.3's backward (color cos 0.99999999, opacity cos 0.99913), not the CUDA kernel. The oracle's
  cos=0 correctness was a DOCUMENTED UNMEASURED GAP, now filled by compositor validation. Per no-rescue and
  cohort-lock, they are NOT remapped, NOT retuned, and NOT counted as PASS.

## External blockers (status 2026-10-03 re-poll)
1. Source access: RESOLVED. Accessible gsplat 1.6.0 source exists (local public_baselines/gsplat + MX gsplat-clean-mx, both v1.6.0 with SerialBatchBwd). Prior “no accessible copy” was incorrect.
2. Exclusive A100: CLEARED. GPU procs dropped 14→3; GPUs 0, 4, 5, 6, 7 are now completely free (0 MiB, 0% util, zero procs). Only GPUs 1,2,3 hold residual root RL-tenant procs. First time exclusive A100s appeared. Transient — must verify exclusivity at run time.
3. ncu profiling: ERR_NVGPUCTRPERM (perf counters root-restricted). Non-blocking for verdict.
4. Toolchain gap: RESOLVED. Prior “CUDA 12 nvcc absent” was WRONG. CUDA 12.8 nvcc + torch 2.9.1+cu128 exist in the higs env (/mnt/storage_pool/liaoyuanjun/higs-13scene-env). PROVEN: gsplat 1.6.0 SerialBatchBwd compiles cleanly to a valid 2.88MB ELF .o (nvcc 12.8.93, -std=c++20, sm_80). The goal-specified nvcc 11.8.89 cannot compile C++20, so the higs CUDA 12.8 toolchain is the viable build path.
5. FORCED DEVIATION: goal-specified runtime (torch 2.7.1+cu118, nvcc 11.8.89) cannot build gsplat 1.6.0 (C++20). No single nvcc supports both C++20 and cu118. Viable path = higs env (torch 2.9.1+cu128, CUDA 12.8). Host/GPU/sm_80 unchanged; only torch+CUDA versions differ.

## Resolution path
HGR0/ABC0 could become PASS only if ALL of: (a) the /root source lock is exposed (to run the real
optimization on-device), (b) an exclusive A100 window opens, AND (c) a real (non-cost-model)
implementation shows the realizable speedup clears the gates (current cost-model realizable estimates
are slowdowns). Compositor correctness is now confirmed faithful (color cos 0.99999999, opacity
cos 0.99913), so the correctness sub-gap is closed; the remaining gaps are the real implementation +
authoritative timing. Until then the program stays NOT_ELIGIBLE. SGOR0/PSR0/HBR0 need no further work.

## Artifacts
- EXACT_OPT_DEDUP_MASTER_VERDICT.json
- _sgor0/SGOR0_DILUTION_ANALYSIS.md, SGOR0_DILUTION_RESULT.txt, SGOR0_STATUS.json
- MX: /mnt/storage_pool/liaoyuanjun/eb0_exact_backward_oracle_mx/{HGR0,ABC0,HBR0,EB0_COMPOSITION}_VERDICT/ORACLE.json
- COMPOSITOR_VALIDATION_RESULT.txt (workspace) + MX mx_compositor_validate2.py, COMPOSITOR_VALIDATION_RESULT.txt

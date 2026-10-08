# COMAX0 E2-prep CUDA Correctness Anchor — REPORT

run_id: e2_correctness_prep_20261006_r3
ts_utc: 2026-10-06T02:09:56Z
host: bms-39468022-001 (MX, 8x A100-PCIE-40GB sm_80)
gpu_uuid: GPU-25086196-7911-cc05-1452-1885d070d710 (index 2)
guard: strict frozen guard via scripts/run_on_mx.py (allowed [1-5], protected [0,6,7]; idle util<=2% AND mem<=512MiB for 20s). NOT the shared-GPU amendment.

## OBSERVED
- Fused dense CUDA kernel vs trusted comax_scan oracle: max_abs RGB = 1.192e-07, p99 = 5.96e-08, p50 = 0.0, alpha_max_abs = 5.96e-8, numel = 768. PASS (<= 2e-5).
- Envelope fast path vs dense fused kernel: max_abs = 0.0 (bit-exact). PASS.
- Envelope vs oracle: max_abs = 5.96e-08. Dense vs oracle: max_abs = 5.96e-08. PASS.
- Dense counters [K_total, rejected, groups_max, hash_fallback] = [2048, 859, 4, 0].
- Envelope counters = [1024, 1024, 1189, 0, 0, 4, 5, 0].
- Stage wall time (correctness fixture only, NOT a timing result): 5.07 s, most of which is CUDA context init.

## EVIDENCE
- run/E2_CORRECTNESS_SUMMARY.json: all_pass=true, dense_all_pass=true, envelope_all_pass=true.
- run/COMAX_CUDA_CORRECTNESS.json: dense checks + counters_last.
- run/COMAX_CUDA_CORRECTNESS_ENVELOPE.json: envelope vs dense vs oracle checks + counters.
- run/GUARD.json: proof (runner_pid, gpu_uuid, idle_samples, target_config_sha256).
- run/COMMAND.json, run/RUN_STATUS.json (scientific_gate_pass=false), run/console.log.
- SOURCE_MANIFEST.sha256 + MANIFEST.sha256 (all verify).

## INFERENCE
- The production fused CUDA dense kernel and the envelope fast path are numerically correct vs the trusted comax_scan oracle on the preregistered synthetic fixture, well inside the 2e-5 forward tolerance (actual ~1e-7).
- The envelope path is bit-exact vs the dense kernel, consistent with the design claim that bound rejection skips only a_eff==0 work.
- This is a NECESSARY correctness anchor for the E2 gate. It is NOT SUFFICIENT: E2 still requires clean guarded timing, real-room correctness, and the reducer's strict E2 evidence set.

## HYPOTHESIS
- The watcher (watch_and_run.sh) never produced this anchor because it launched the dense harness directly, so require_guarded_process() raised immediately (no COMAX0_GUARD_PROOF), and separately the CPU-written fixture/background tensors are not GPU-safe. The wrapper run_e2_correctness.py fixes both without editing production source or the COMAX algorithm.

## NOT_MEASURED
- No wall-clock speed, CUDA kernel speed, throughput, memory-bandwidth, or E2/E3 production cost (performance_evidence_valid=false, timing_run=false).
- No room-scale correctness, no room rejection rate rho, no traversal complexity Q, no divergence/occupancy counters.
- The synthetic-fixture rejection (859/2048 = 41.9%) is a kernel counter on 1 tile / 8 gaussians / 4 groups; it is NOT the room work-weighted rho and must not be read as such.

## CONCLUSION
- E2-prep CUDA-vs-oracle correctness anchor: PASS (numerics only).
- Master verdict UNCHANGED: INCONCLUSIVE_BUDGET (this anchor does not advance the reducer; scientific_gate_pass=false).
- Next legitimate step toward E2 remains: clean guarded GPU + real-room correctness + strict reducer evidence, in the original order. Do NOT run E2 timing on this synthetic fixture and do NOT reinterpret this as a speed result.

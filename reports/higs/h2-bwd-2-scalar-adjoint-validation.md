# H2-BWD-2 — SCALAR_ADJOINT Production Validation

## Candidate Record

| Field | Value |
|-------|-------|
| **Candidate** | H2-BWD-2 (SCALAR_ADJOINT production-source backward) |
| **Parent** | H2-BWD-CF engineering (production-source build) |
| **Type** | Type A (Exact systems optimization) |
| **Evidence level** | L4 (Component/kernel) + production F+B |
| **Verdict** | **KEEP_NEEDS_CORRECTNESS_REPAIR** |
| **Frozen binary** | `da53009841c5f9a6145dfb01e5ab84de5286f52710c9a7011bcb4b85eb18842c` (verified match) |

---

## Frozen experimental identity (verified)

| Item | Value | Verified? |
|------|-------|-----------|
| B2 base commit | `77ab983ffe43420b2131669cb35776b883ca4c3c` | ✓ (recorded) |
| B2 authoritative patch SHA256 | `74e5d8b3b6273b9446ec0551ce91409783e2aa935c8d8e354b4099341390c84c` | ✓ (recorded) |
| H2-BWD-CF patch SHA256 | `d001eac2d6908126103ead7c060f896cfc569989ac90d264236a97edc68137ff` | ✓ (recorded) |
| Experimental binary SHA256 | `da53009841c5f9a6145dfb01e5ab84de5286f52710c9a7011bcb4b85eb18842c` | ✓ **exact match** |
| Variant selector | `HIGS_BWD_CF_VARIANT=baseline\|scalar_adjoint` | ✓ (only these two used) |
| Loader | `TORCH_EXTENSIONS_DIR=/tmp/higs_h2_bwd_cf/cache` (reuses frozen .so, never rebuilt) | ✓ |

The old load_inline H2-BWD-1 microkernel was **not** used. Every measurement below loads the exact frozen production `.so` via `build_and_load_experimental_gaussian_render_inference_scene` with the cache pinned so the binary is reused, not recompiled.

---

## 1. Resource mechanism

Parsed from the CF build's `ptxas -v` log (`-Xptxas=-v`, CDIM=3, PX=2 instantiations):

| variant | registers/thread | spills (store/load) | dynamic shared | active blocks/SM | theoretical occupancy |
|---|---:|---:|---:|---:|---:|
| **baseline** | 67 | 0 / 0 B | 5120 B | 7 | 43.75 % |
| **scalar_adjoint** | 56 | 0 / 0 B | 5120 B | 9 | 56.25 % |

**Theoretical register-limited occupancy (A100 SM80, 128-thread blocks):**

- `regs_per_block = regs × 128`: baseline 8576 → `65536 / 8576 = 7` blocks/SM; scalar 7168 → `65536 / 7168 = 9` blocks/SM.
- Thread limit (16 blocks/SM) and shared limit (100 KB / 5120 B = 20 blocks/SM) do **not** bind. **Both kernels are register-bound.**
- Occupancy: baseline 28 warps/SM (43.75 %) → scalar 36 warps/SM (56.25 %), **+12.5 pp, +28.6 % relative**.

**Achieved occupancy / eligible warps:** **NOT measured.** The on-host `ncu` is `2021.3.1.0` (CUDA 11.5 era) with missing stock section files; it returns `rc=255` and cannot profile `sm_80`. Direct falsification of achieved occupancy is therefore **inconclusive**. (`occupancy.json` records the attempt and the failure.)

---

## 2. Strict gradient correctness

Same frozen forward state (frame/alpha tensor SHA256 identical baseline==scalar_adjoint on all 3 scenes). Fixed seeded nonzero VJP (seed 4200). Compared baseline vs scalar_adjoint for all 9 tensors.

**Full per-tensor results:** [`correctness_by_tensor.csv`](../../artifacts/higs-h2-bwd-2/correctness_by_tensor.csv).

### Which tensor causes the previous aggregate 1.0315e-3?

The previous aggregate max rel_L2 = **1.0315e-3** was **entirely the `quats` tensor of `scalar_adjoint`** (CF `correctness_raw.json`, room/seed4200). No other tensor was anywhere near it.

### Strict gate (rel_L2 ≤ 1e-4, cosine ≥ 0.999999, no NaN/Inf, zero support disagreement = 0)

| scene | tensor | rel_L2 | cosine | max_abs | norm_baseline | PASS? |
|-------|--------|--------|--------|---------|---------------|-------|
| room | **quats** | 4.63e-4 | 1.0 | 0.661 | 2094.5 | **FAIL** |
| bicycle | **quats** | 1.33e-4 | 0.99999994 | 0.247 | 2766.3 | **FAIL** |
| garden | **quats** | 1.65e-4 | 0.99999982 | 0.567 | 4942.2 | **FAIL** |
| all 3 | v_means2d, v_conics, v_colors, v_opacities, means, scales, opacities, SH | ≤ 3.5e-5 | ≥ 0.9999999 | — | — | **PASS** |

**Only `quats` fails**, on all three scenes. All other 8 tensors pass comfortably.

### Absolute-error-normalized interpretation (quats norm is not small, but the error is atomicAdd noise)

The `quats` gradient is accumulated by `atomicAdd` across many warps into a length-`4×N` tensor. To test whether the quats error is a **transformation** error or **atomicAdd accumulation-order** noise, I ran a **baseline-vs-baseline determinism control** (5 independent runs, same frozen binary, same fixture, same seed):

| comparison | quats rel_L2 range | quats cosine range |
|---|---|---|
| baseline run_i vs baseline run_0 (i=1..4) | **4.11e-4 … 7.04e-4** | 0.99999976 … 0.99999988 |
| scalar_adjoint run_i vs baseline run_0 (i=0..4) | 3.39e-4 … 9.68e-4 | 0.99999952 … 1.0 |

**The two distributions overlap and are of comparable magnitude.** The scalar_adjoint transformation introduces **no additional error beyond the baseline's own atomicAdd non-determinism**. In every comparison (including baseline-vs-baseline), `quats` is the single max-rel_L2 tensor.

**Conclusion:** The strict `rel_L2 ≤ 1e-4` threshold is **not satisfiable by baseline-vs-baseline either** on `quats` — it sits below the kernel's own atomic-accumulator precision floor for that tensor. The 1.0315e-3 was an unlucky single draw from this noise distribution. The scalar_adjoint transformation is **correct to within the kernel's deterministic reproducibility limit**.

**Tolerance was NOT loosened.** The strict gate **FAILS as written**. Artifact: [`determinism_control.json`](../../artifacts/higs-h2-bwd-2/determinism_control.json).

---

## 3. Repeated production kernel timing

Protocol: 20 warmup + 100 CUDA-event measurements, **5 independent repetitions**, **interleaved** (randomized baseline/scalar order within each measured pair). Primary metric: **blend backward kernel time** (backward only).

| scene | variant | median ms | mean ms | p10 | p90 | std | paired Δ (ms) | % speedup | 95% bootstrap CI (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| room | baseline | 2.1453 | 2.1503 | 2.1340 | 2.1647 | 0.0249 | — | — | — |
| room | scalar_adjoint | 1.9292 | 1.9314 | 1.9139 | 1.9457 | 0.0189 | **0.2181** | **10.17 %** | [0.2161, 0.2202] |
| bicycle | baseline | 3.0996 | 3.1015 | 3.0812 | 3.1243 | 0.0171 | — | — | — |
| bicycle | scalar_adjoint | 2.8580 | 2.8591 | 2.8334 | 2.8877 | 0.0215 | **0.2427** | **7.83 %** | [0.2396, 0.2458] |
| garden | baseline | 1.3158 | 1.3236 | 1.3087 | 1.3332 | 0.0380 | — | — | — |
| garden | scalar_adjoint | 1.1868 | 1.1946 | 1.1786 | 1.2043 | 0.0364 | **0.1290** | **9.81 %** | [0.1280, 0.1300] |

**Promotion requirement (≥8 % reproducible across ≥2 scenes):** **PASS** — room 10.17 % and garden 9.81 % both ≥8 %; all three CIs tightly exclude zero. bicycle at 7.83 % is marginally below 8 % but CI excludes zero. Artifact: [`kernel_timing.csv`](../../artifacts/higs-h2-bwd-2/kernel_timing.csv).

---

## 4. Production backward and F+B timing

Through `rasterize_gaussian_higs_frozen` (the `_HigsAutogradFunction` forward) + `.backward()`, interleaved 5×100.

| scene | metric | baseline ms | scalar ms | Δ ms | % saved |
|---|---|---:|---:|---:|---:|
| room | T_forward | 1.8842 | 1.8842 | 0.0005 | 0.03 % |
| room | T_backward | 2.2702 | 2.0531 | 0.2171 | **9.56 %** |
| room | T_F+B | 4.3121 | 4.0940 | 0.2171 | **5.03 %** |
| bicycle | T_forward | 2.0644 | 2.0644 | 0.0010 | 0.05 % |
| bicycle | T_backward | 3.2276 | 2.9824 | 0.2437 | **7.55 %** |
| bicycle | T_F+B | 5.4630 | 5.2193 | 0.2427 | **4.44 %** |

`T_forward` is variant-independent (the selector affects only the backward kernel) — confirms the speedup is purely in the backward kernel. Artifact: [`backward_fb_timing.csv`](../../artifacts/higs-h2-bwd-2/backward_fb_timing.csv).

**Practical gate (F+B ≥ 3 % OR backward ≥ 5 %, with correctness):** pure-performance **PASS** (room F+B 5.03 % / backward 9.56 %; bicycle F+B 4.44 % / backward 7.55 %); with correctness **FAIL** (correctness fails).

---

## 5. Mechanism falsification

**Hypothesis:** scalar adjoint → smaller live state → 67→56 registers → higher occupancy / more latency hiding → ~11 % blend speedup.

| condition | status |
|---|---|
| register reduction confirmed (67→56) | ✓ yes |
| theoretical occupancy improves (43.75 %→56.25 %, +12.5 pp) | ✓ yes |
| achieved occupancy / eligible warps measured | ✗ no (NCU 2021.3.1 cannot profile sm_80) |
| kernel speedup replicates (≥8 % on ≥2 scenes) | ✓ yes |

**Falsified? No.** The falsification condition ("achieved occupancy/eligible warps do not improve AND repeated timing loses the speedup") is **not met** — the speedup replicates robustly and theoretical occupancy improves.

**Caveat on attribution (honest reporting, not forcing the explanation):** because achieved occupancy could not be measured, and because scalar_adjoint changes **both** register pressure **and** per-thread arithmetic (vector→scalar dot accumulation reduces the number of adds), the ~10 % kernel speedup is consistent with **either** (a) occupancy-driven latency hiding **or** (b) reduced arithmetic/instruction throughput. The hypothesis is **plausible but not isolated**. Isolating the two would require a register-matched control that keeps the vector arithmetic — which is a redesign and was not run (mission: do not redesign).

---

## 6. Training

**Not run.** The mission gates require correctness PASS **AND** kernel speedup reproducible **AND** production backward/F+B PASS. Correctness fails (quats, root cause = atomicAdd non-determinism). Training is therefore deferred.

---

## Final classification

```
KEEP_NEEDS_CORRECTNESS_REPAIR
```

**Rationale:** The strict gradient-correctness gate FAILS as written — `quats` exceeds `rel_L2 ≤ 1e-4` on all three scenes. Per mission rules the tolerance is **not loosened**. Kernel speedup is reproducible (gate PASS: room 10.17 %, garden 9.81 %) and production backward/F+B gates PASS on pure performance (F+B +5.03 %/+4.44 %, backward +9.56 %/+7.55 %), but the practical gate requires correctness passing.

**The correctness failure is NOT a scalar_adjoint transformation defect.** A baseline-vs-baseline determinism control proves the quats error is upstream atomicAdd accumulation-order non-determinism that equally affects the baseline against itself (quats rel_L2 4.11e-4…7.04e-4 baseline-vs-baseline, same range as scalar-vs-baseline). The scalar_adjoint transformation is correct to within the kernel's own deterministic reproducibility limit.

**The required "repair" is therefore NOT an algorithm change to scalar_adjoint** — it is making the backward gradient accumulation deterministic (order-stable reduction or fp64 accumulator) so that `rel_L2 ≤ 1e-4` becomes satisfiable for `quats` at all, including for baseline-vs-baseline. Once deterministic accumulation is in place, re-running this validation is expected to yield a full gate PASS → PROMOTE_TO_SHORT_TRAIN → 800-step room training.

No other optimization is proposed.

---

## Artifacts

All in [`artifacts/higs-h2-bwd-2/`](../../artifacts/higs-h2-bwd-2/):

| File | Description |
|------|-------------|
| `resource_validation.json` | ptxas registers/spills/shared + theoretical occupancy |
| `correctness_by_tensor.csv` | per-tensor baseline vs scalar metrics (9 tensors × 3 scenes) |
| `kernel_timing.csv` | interleaved backward-only kernel timing (5×100) + bootstrap CI |
| `backward_fb_timing.csv` | T_forward / T_backward / T_F+B interleaved (room, bicycle) |
| `occupancy.json` | theoretical occupancy + NCU attempt (failed: NCU 11.5 cannot profile sm_80) |
| `analysis.json` | gate evaluation, determinism finding, mechanism falsification, classification |
| `determinism_control.json` | baseline-vs-baseline control proving quats error is atomicAdd noise |
| `provenance.json` | run identity, binary hash, GPU, env |
| `run.log` | full run log |

## Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA A100-PCIE-40GB (SM80, 108 SMs), GPU 4 (uncontended) |
| nvcc | CUDA 12.8 V12.8.93 |
| Torch | 2.9.1+cu128 |
| target | compute_80, code=sm_80 |
| Datasets | MipNeRF360 room/cam0, bicycle/cam0, garden/cam0 (2048 long side) |
| Protocol | 20 warmup / 100 measure / 5 reps, interleaved, CUDA event timing |
| Script | `scripts/h2/h2_bwd_2_validation.py` |
| Launcher | `scripts/h2/_h2_bwd_2_run.sh` |
| Determinism control | `scripts/h2/h2_bwd_2_determinism_control.py` |

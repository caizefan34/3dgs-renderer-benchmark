# P2-1A — Native Hierarchical Forward (FP32 Adapter Build + Runtime Gate)

**Status: `P2_1A_CORRECTNESS_FAIL_AT_ENTRY`** — the as-built prototype (`1799facc`) was put on a GPU the moment a correctness-grade probe was possible, and **cannot execute a single forward render**. Both Python-reachable forward entry points throw `RuntimeError: expected scalar type Half but found Float` in host launch code, deterministically, before any ported kernel runs. Per the gate protocol (§5), performance work is stopped and the correctness failure is returned. No timing, hierarchy-counter, P2-1C-metadata, or runtime-resource measurement of the ported kernels is possible against this artifact. **P2-1C remains NOT AUTHORIZED.**

The build-phase record below is preserved verbatim as provenance; the runtime-gate execution follows it.

---

## Part I — Build provenance (frozen CUDA 12.8, compile-only, unchanged)

- **Worktree**: `/mnt/storage_pool/liaoyuanjun/higs_p2_1a_worktree` at frozen commit `77ab983ffe43420b2131669cb35776b883ca4c3c`
- **Frozen env**: `/mnt/storage_pool/liaoyuanjun/higs-13scene-env` (torch `2.9.1+cu128`, CUDA 12.8, python 3.10.21)
- **nvcc**: `release 12.8, V12.8.93` (host nvcc 11.5 deliberately **not** used)
- **Arch**: `TORCH_CUDA_ARCH_LIST=8.0` → `-gencode=arch=compute_80,code=sm_80`
- **Flags**: `-Xptxas -v -use_fast_math --expt-relaxed-constexpr -std=c++20 -O3`
- Extension name: `experimental_gaussian_render_inference_scene_cuda`
- **Build result**: `LINK_SUCCESS`, undefined symbols: none, `MODULE_LOADED` → [`build_log.txt`](../../artifacts/higs-p2-1a/build_log.txt), [`build_provenance.json`](../../artifacts/higs-p2-1a/build_provenance.json)

### Files adapted (FP32 port, 6 files — scope that later proved incomplete)

`IntersectCommon.h` (float4 conic decode), `IntersectMTFused.cu/.h` (float binning fences/host, float tile buffer), `MacroTileRasterize.cu/.h` (full FP32 raster + post-blend), `GaussianRenderInferenceScene.cu` (state/validation/background to `kFloat`).

**Not adapted**: `Projection.cu`, `SphericalHarmonics.cu`, `ext.cpp` — the consequence is recorded in Part II.

### Artifact SHAs (re-verified on the runtime gate day, after all probes)

| artifact | SHA256 |
|---|---|
| `gsplat_cuda_p2_1a.so` (TEST) | `1799facc0fa3c30375aad4edeff28b091ce35891cbb794f467c0984e76e14089` |
| patch | `ab860d771b4adb1afd72615ecaa5342d020de43d421ff653990f083cd16b496c` |
| BASE composed .so (C0 V3) | `7ca1c6bf6c8e4307ecb8fcdbcaf2953bf95305d2c9814f84f3fb5130859301f6` |
| BASE core .so | `361b216bcc11609a0ebb8fb44ad2e0c6170948112b6294e85123df45541c8c98` |

### Static-resource audit (compile-only, stands as stated)

raster regs 64, static smem 47,760 B, spills `<8>` 32/32 B, `<16>` 72/76 B; joint static residency 3 CTAs/SM, 46.875% theoretical occupancy (registers bind; shared ties at FP32) — **not** 1 CTA/SM. See [`fp32_resource_delta.json`](../../artifacts/higs-p2-1a/fp32_resource_delta.json), [`occupancy_corrected.json`](../../artifacts/higs-p2-1a/occupancy_corrected.json), [`spill_attribution.json`](../../artifacts/higs-p2-1a/spill_attribution.json).

---

## Part II — Runtime gate execution (2026-09-22)

### 1. GPU reconnaissance and selection

Live state at gate open (8× A100-PCIE-40GB, driver 595.71.05): GPU0 2.3 GB free (memory-starved); GPU1 97%; GPU2 40%; GPU3 30% w/ vLLM; GPU4 39%; GPU5 40%; GPU6 99%; GPU7 97% w/ vLLM. **No clean GPU.** Per §1 of the protocol, correctness work proceeded on the shared GPU with the most headroom: **GPU index 3** (22.9 GB free). No timing was attempted anywhere (C0 publication timing is frozen and was not rerun).

### 2. Entry executability matrix (the first runtime measurement of the as-built artifact)

Synthetic SH3 scene (N=64, means_planar [3,N] fp32, qso_packed [N,8] fp16, colors_packed [N,16,3] fp16, 256×192, tile 16, sh_degree 3, compression NONE) — exercises the exact `state.sh_coeffs_per_channel == 16` fused-projection path every real scene would take. One process per (.so, mode) pair (both .so register `TORCH_LIBRARY(experimental)`). Full raw log: `/mnt/storage_pool/liaoyuanjun/higs_p2_1a_results/entry_smoke_matrix.log`.

| variant | entry | result |
|---|---|---|
| TEST `1799facc` | `GaussianInferenceRenderer.render` | **FAIL** — `RuntimeError: expected scalar type Half but found Float` |
| TEST `1799facc` | `torch.ops.experimental.gaussian_render_inference_only` | **FAIL** — same error |
| TEST `1799facc` | `higs_gatherless_projected_producer` (F9 binding) | **PASS** — all-FP32 outputs, 64/64 positive radii |
| C0 `7ca1c6bf` (control) | `GaussianInferenceRenderer.render` | **PASS** — rgbt [1,192,256,4] fp16, rgb_mean 0.009954, T_mean 0.980294 |
| C0 `7ca1c6bf` (control) | `gaussian_render_inference_only` op | **PASS** — renders [192,256,3] fp32, alphas [192,256,1] fp32 |
| C0 `7ca1c6bf` (control) | F9 producer | **PASS** — identical contract |

The invocation format is proven correct by the controls; the failure is specific to the P2-1A FP32 port. The F9 producer binding inside the very same TEST .so works, isolating the break to the projection→state handoff.

### 3. Root cause — incomplete FP32 port (producer launch ABI not converted)

1. The patch (6 files) converted `state->conics`/`state->colors` to **float** and the whole downstream hierarchy (IntersectMTFused, MacroTileRasterize) to FP32.
2. `render()` passes those float state buffers to `higs::launch_projection_sh_fused_kernel` (SH3/NONE path) / `launch_projection_fwd_kernel` (all other paths) — but **`Projection.cu` was never patched**.
3. Its host launch does `conics.data_ptr<at::Half>()` / `colors.data_ptr<at::Half>()`; `at::Tensor::data_ptr<T>` type-checks and throws — before any kernel launch.
4. The device side agrees: ptxas symbols for `higs::projection_fwd_kernel<…>` end with `__half*` conics and `__half*` colors. Absent the host check the kernel would type-confuse half writes into float buffers; the throw is the safe failure path.
5. `gaussian_render_inference_only` builds state through the same creator and dispatches into the same path → identical failure.

**Correction of the build-phase audit**: [`abi_check.json`](../../artifacts/higs-p2-1a/abi_check.json) listed `Projection.cu` under "remaining half" with the justification that these components "feed FP32 float state.colors". That is incorrect — the unpatched projection launch **cannot** feed the FP32 state at all; it hard-fails. The static ABI audit covered the raster/binning/post-blend launch surface but omitted the projection launch surface against the patched state buffers. This is exactly the class of defect the "no correctness claim from unexecuted kernels" caveat anticipated.

### 4. Secondary structural finding — the intended TEST composition is not Python-reachable even past the dtype fix

`ext.cpp` is **byte-identical** between the C0 and P2-1A worktrees. `IntersectMTFused::execute` / `::rasterize` have **no pybind bindings**. The only Python forward entries are `render()` / `gaussian_render_inference_only`, which run their **own internal projection from the FP16-packed inference layout** (`qso_packed` [N,8] half, `colors_packed` [N,16,3] half) rather than consuming the F9 gatherless producer's FP32 output. Two consequences for any repair:

- a dtype-only repair leaves the entry consuming **FP16-quantized inputs** while BASE consumes FP32 masters — Gate C would then measure FP16 input quantization, and `ALGEBRAIC_EXACT_FP_REASSOCIATED` is unreachable via that entry;
- the task's TEST stage decomposition (**shared F9 producer** → macro partition → … → post-compose) requires new bindings (option B below).

### 5. Gate outcomes

| gate | outcome |
|---|---|
| A — structural equivalence | **BLOCKED** — TEST pair generation unreachable ([structural_equivalence.json](../../artifacts/higs-p2-1a/structural_equivalence.json)) |
| B — ordering equivalence | **BLOCKED** — segmented-sort output unreachable ([ordering_equivalence.json](../../artifacts/higs-p2-1a/ordering_equivalence.json)) |
| C — RGB/alpha correctness | **FAILED AT ENTRY** — no TEST output tensor exists ([forward_correctness.json](../../artifacts/higs-p2-1a/forward_correctness.json)) |
| §8 hierarchy counters | **BLOCKED** — no ported kernel ever launched |
| §9 P2-1C metadata | **BLOCKED** — P2-1C **NOT AUTHORIZED** ([p2_1c_metadata_validation.json](../../artifacts/higs-p2-1a/p2_1c_metadata_validation.json)) |
| §10 runtime resource | static figures stand; runtime confirmation **BLOCKED** ([runtime_resource.json](../../artifacts/higs-p2-1a/runtime_resource.json)) |
| §11–12 authoritative timing | **NOT RUN** — requires an executable TEST forward ([timing_summary.json](../../artifacts/higs-p2-1a/timing_summary.json), [raw_forward_timing.csv](../../artifacts/higs-p2-1a/raw_forward_timing.csv) header-only, zero fabricated samples) |
| §13 stage decomposition | **NOT RUN** + static reachability finding ([stage_breakdown_runtime.json](../../artifacts/higs-p2-1a/stage_breakdown_runtime.json)) |
| §14 GPU interference | **NOT APPLICABLE** — no timing attempted or possible; the failure is a host-side type error, independent of GPU load |

Full gate state and evidence index: [`runtime_gate.json`](../../artifacts/higs-p2-1a/runtime_gate.json), [`runtime_provenance.json`](../../artifacts/higs-p2-1a/runtime_provenance.json).

### 6. Final answer to the §19 questions

1. **GPU used / live load**: GPU 3 (A100-PCIE-40GB, shared with a vLLM worker; 30% SM at probe time) for correctness probes only; no clean GPU existed (live table in §II.1). Moot for the verdict — the failure is deterministic host-side.
2. **Structural equivalence**: not evaluable — TEST forward not executable.
3. **Ordering equivalence**: not evaluable — TEST forward not executable.
4. **RGB/alpha correctness**: **FAILED AT ENTRY** — TEST produces no image; the identical harness renders correctly on the C0 control .so.
5. **Hierarchy compression**: unmeasured (blocked).
6. **Dead-group fractions**: unmeasured (blocked); room ~20% / bicycle ~30% / garden ~20% remain unmeasured hypotheses, explicitly **not** recorded as fact.
7. **P2-1C metadata cost**: unmeasured (blocked).
8. **Runtime raster/resource observations**: only the F9 producer binding executed (43 regs, 0 spills); the ported raster never ran, so regs/smem/spill figures remain compile-time only; CUDA context and module load behave normally (not an environment problem).
9. **room BASE→TEST**: not measurable.
10. **bicycle BASE→TEST**: not measurable.
11. **garden BASE→TEST**: not measurable.
12. **Paired timing**: none (no samples; nothing fabricated).
13. **Five-rep stability**: none (no samples).
14. **Stage breakdown**: not runnable; additionally the intended TEST composition (shared F9 producer → native hierarchy) is not Python-reachable in this build (no bindings for `IntersectMTFused::execute/rasterize`).
15. **Authoritative vs diagnostic timing**: neither — timing impossible against this artifact; the correctness failure is GPU-state-independent.
16. **Classification**: **`P2_1A_CORRECTNESS_FAIL_AT_ENTRY`** — none of STRONG / MARGINAL / RESOURCE_LIMITED / WEAK is assignable (all require an executable TEST forward; §16's exception likewise requires timing evidence).
17. **P2-1C authorization**: **NOT AUTHORIZED** (requires correctness PASS + STRONG).
18. **Exact next action**: authorize exactly one repair round, then rerun this gate end-to-end on the new artifact:
    - **Option B (recommended, contract-true)**: add pybind bindings for a native-hierarchy forward entry consuming the F9 gatherless producer's FP32 outputs (means2d, depths, conics[N,3]+opacity, colors, visible mask) driving `IntersectMTFused::execute + rasterize`; this matches the mandated stage decomposition and enables the exact FP32 comparison the gates target.
    - **Option A (smoke-level intermediate only)**: convert `Projection.cu` (and `SphericalHarmonics.cu` color emission) to float conics/colors outputs — makes `render()`/op executable, but leaves FP16-quantized inputs, so `ALGEBRAIC_EXACT_FP_REASSOCIATED` stays unreachable via that entry.
    - Either way: rebuild in the frozen env, record new .so/patch SHAs, rerun §5→§13 in order. The `1799facc` artifact is closed.

### 7. Protocol compliance notes

- "Do not rebuild or modify prototype before first runtime measurement" — **compliant**: first runtime contact was the load+render probe; the .so SHA was re-verified identical after all runs.
- C0 publication timing was **not** rerun; the C0 .so was used only as an executability control, never re-timed.
- No samples, counters, or fractions were fabricated; every blocked artifact states its blocker and points at the evidence.

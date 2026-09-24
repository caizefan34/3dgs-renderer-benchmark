# C1 — P5 Sort Performance Verification

> **Target:** `mx` Linux/A100 environment
>
> **Scope:** P5.1–P5.5 only. No C1 key-layout, quantization, or optimization-mechanism changes were made during this verification attempt.
>
> **Final P5 signal:** **NEGATIVE** (validation attempt)
>
> This is negative for the **current P5 validation attempt** only — the executable CUDA evidence chain was not established. It is **not** a measured runtime claim against C1. The runtime impact of the reduced sort key width remains unvalidated because the target CUDA backend could not be built reproducibly.

---

## 1. Target Environment Fingerprint

| Field | Value |
|---|---|
| Hostname | `bms-39468022-001` |
| GPU | NVIDIA A100-PCIE-40GB |
| GPU memory | 40,960 MiB |
| Compute capability | 8.0 (`sm_80`) |
| NVIDIA driver | 595.71.05 |
| Driver-reported CUDA capability | 13.2 |
| System `nvcc` | CUDA 11.5.119 |
| Host compiler | GCC 11.4.0 (Ubuntu 22.04) |
| Python | 3.10.12 |
| PyTorch | 2.7.1+cu118 |
| PyTorch CUDA runtime | 11.8 |
| gsplat | 1.5.3 |

The host exposed eight A100-PCIE-40GB GPUs to `nvidia-smi`. No comparison against prior or different hardware was performed.

---

## 2. P5.1 — Build Status

### Baseline source state

The installed gsplat source was verified as baseline before the rasterizer invocation:

```cpp
const int64_t iid_enc = iid << (32 + tile_n_bits);
isect_ids[cur_idx] = iid_enc | (tile_id << 32) | depth_id_enc;
int64_t isect_id_curr = isect_ids[idx] >> 32;
```

The baseline CUB radix-sort bit-range expression in source is:

```cpp
32 + tile_n_bits + image_n_bits
```

### Build result: failed before baseline forward

The first baseline `gsplat.rasterization(...)` attempt, at 5K Gaussians and 960×540, failed before a rendered output was created. The final captured error was:

```text
SyntaxError: keyword argument repeated: with_sycl
```

in:

```text
/home/liaoyuanjun/.local/lib/python3.10/site-packages/gsplat/cuda/_backend.py
```

Earlier bounded diagnostics on the same environment also established:

1. gsplat 1.5.3 uses a private PyTorch `_jit_compile` call incompatible with PyTorch 2.7.1's signature.
2. A clean JIT rebuild using system CUDA 11.5 and GCC 11.4 failed during gsplat CUDA compilation under C++17 in GCC's `std_function.h` with a parameter-pack error.
3. `ninja` was installed but initially unavailable from the non-interactive PATH; fixing that did not resolve the PyTorch/gsplat or compiler incompatibilities.

No pre-existing compatible conda/mamba environment, Docker/Podman image, or cached gsplat extension binary was found during the bounded probe.

### Preserved remote evidence

| Artifact | Location |
|---|---|
| Baseline build/forward attempt log | `mx:~/c1_p5_baseline_build.log` |
| Baseline runner | `mx:~/p5_baseline_runner.py` |
| Installed source under test | `mx:~/.local/lib/python3.10/site-packages/gsplat/` |

---

## 3. P5.2 — Isolated CUB `SortPairs` Timing

| Requirement | Result |
|---|---|
| Baseline actual key | Source identified; CUDA extension not runnable |
| Baseline actual bit range | `32 + tile_n_bits + image_n_bits` in source |
| C1 actual key and bit range | Not built or run |
| CUDA event measurement | Not executed |
| 30 warmups / 100 samples / 3 repeats | Not executed |
| Mean / median / standard deviation / min / max | N/A |
| `sort_speedup` | N/A |

No sort speedup was inferred from the source-level key-width reduction.

---

## 4. P5.3 — Forward Timing

| Requirement | Result |
|---|---|
| Baseline total forward | Not available: baseline extension did not build/run |
| C1 total forward | Not attempted after baseline prerequisite failure |
| Baseline/C1 sort fraction | N/A |
| Real-scene timing | Not executed |
| End-to-end speedup | N/A |

No cross-hardware timing comparison was performed.

---

## 5. P5.4 — CUDA Correctness Smoke Check

| Check | Result |
|---|---|
| Baseline / C1 intersection count | Not available |
| Sorted item count | Not available |
| Tile grouping and offsets | Not available |
| Baseline / C1 rendered image | Not available |
| Real CUDA PSNR / SSIM / max / mean pixel error | Not available |

The earlier Python tile-composite simulation is explicitly not formal P5 correctness evidence.

---

## 6. P5.5 Decision

## NEGATIVE (validation attempt) / INCONCLUSIVE (overall C1 hypothesis)

The accessible target failed P5.1's prerequisite: a reproducible, runnable baseline gsplat CUDA backend was not established. Therefore CUDA-event `SortPairs` timing, forward timing, rendered-image correctness checks, and speedup calculations cannot be reported honestly.

**Critical distinction:** This negative signal applies **only to the current validation attempt on the `mx` Linux/A100 environment**. It does **not** falsify C1's runtime hypothesis. The runtime impact of the reduced sort key width remains unvalidated because the target CUDA backend could not be built reproducibly.

### P6 disposition — FROZEN / ARCHIVED-PENDING

C1 is frozen pending a future environment with a known-compatible gsplat/PyTorch/CUDA/compiler combination. The C1 patch itself is complete and ready: source-audited, five-line IntersectTile.cu change plus corresponding offset-shift and CUB end_bit adjustments. No further design work, quantization theory, or optimization mechanism addition is warranted.

Supported conclusion:

> C1 (16-bit depth-key narrowing) is a VALIDATION-BLOCKED candidate. The reduced sort key width reduces the CUB end_bit parameter by 16 bits (source-verified), which reduces the CUB sort key width from 46 to 30 bits (global) and from 45 to 29 bits (segmented). The actual runtime effect of this reduction is unmeasured because a reproducible CUDA baseline could not be established on the accessible target. The hypothesis remains open: the experimental measurements required to confirm or falsify a measurable speedup were not obtained, and no claim of performance gain or loss should be made from the available evidence.

---

## 7. Re-entry Condition

Reopen P5 only after an environment provides all of:

1. A reproducible unmodified gsplat baseline CUDA build and rasterizer forward pass;
2. A compatible PyTorch, gsplat, CUDA toolkit, and host compiler combination;
3. Separate baseline and C1 build artifacts;
4. CUDA-event timing capability for 30 warmups, 100 measured samples, and 3 repeats.

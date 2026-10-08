# A0-P10 Preflight — Remote Runbook (EPIC-05, A100)

Scope (frozen by user directive, **REVISED per lead review**): **one** frozen cell —
`room @ 25K`, frozen training camera index **19** (`DSCF4819`) — top-32 **native
bucket objects** (tile × batch) ranked **only** by native accepted-hit count.
Export exact FP32 `q_alpha` + primary `q_color` (real 3-channel) [+ secondary
`w_color`], run Stage-0 identity (with kernel byte-identity + camera + checkpoint
asserts), **preflight-sanity** checks, and the oracle-SVD/DCT analyzer, then emit
the **frozen mechanical P10 verdict** (oracle SVD only). **No** DCT/VJP
reconstruction, **no** bicycle/20K, **no** full campaign, **no** production CUDA.

This is a **preflight** to de-risk the full A0-v2 campaign. The P10 checks are
**preflight sanity only** — they do **not** claim full A0-v2 native-gradient
parity (that belongs to formal A0). The full A0-v2 verdict (all 4 cells +
reconstructed-VJP + cost gates) is out of scope here.

---

## 0. What is in the kit

```
A0_P10_PREFLIGHT_KIT/
  a0preflight/
    lib.py                    shared: frozen paths, bootstrap_c0, render, loss,
                              exact C0 blend-bwd WALK over native (tile,batch) buckets
                              (T init T_final, q_color=fac*v_render_colors, corrected
                              Common.h constants), dense coefficient builder,
                              2D DCT-II basis (bit-identical to the frozen analyzer),
                              frozen mechanical P10 verdict
    oracle_analyze.py         oracle-SVD / DCT structural analyzer (frozen semantics;
                              joint Q_color from the REAL 3-channel q_color tensor)
    a0p10_identity.py         Stage 0 remote identity / hash verification (incl.
                              kernel byte-identity + cam-19 img_name + room/25K sha)
    a0p10_run.py              one-shot driver (identity -> native graph+raw-frame
                              adjoint -> bucket census -> dump -> preflight-sanity ->
                              analyze -> verdict)
  A0_EXACT_BACKWARD_FACTORIZATION.md   Stage 1 exact algebra (q_alpha / q_color)
  RUNBOOK_A0_P10.md          this file
  tests/_test_p10.py         local synthetic verification (not run remotely)
  MANIFEST_A0_P10.json       sha256 of every kit file
```

## 1. Prerequisites (must already be true on EPIC-05)

The frozen P12/C0 stack is already present (verified by prior P12/MG0/Q0 identity
records). The runbook does **not** build or install anything:

- Frozen gsplat env: `/root/miniforge3/envs/gsplat/bin/python`
- C0 worktree: `/root/epic05_frozen/higs_c0_final30k_worktree`
- C0 ext `.so`: `/root/epic05_build_cache/c0_ext/.../experimental_gaussian_render_inference_scene_cuda.so`
  (P12 publication lineage `d26cccf4…`)
- Core `.so`: `/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so`
  (frozen C0 lineage `935ea6c0…`)
- Fused-SSIM loss backend: `…/fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so`
  (P11-locked `dd6a4c79…`)
- Checkpoint: `/root/epic05_n1/runs/n1snap_c0f_s42_room/ckpt_25000.pt`
  (authoritative sha `b31e5a6b…`, asserted by Stage 0)
- Dataset: `/root/3dgs-renderer-benchmark/data/official/mipnerf360/room` (1080p)
- Authoritative kernel sources for byte-identity: the kit references
  `_moonshot/HigsNativeBackward.cu` + `_moonshot/RasterizeToPixels3DGSDevice.cuh`
  (byte-identical to the remote frozen C0 files).

## 2. Deploy the kit

Copy the kit **and the `_moonshot/` kernel sources** to the A100 host:

```bash
scp -r A0_P10_PREFLIGHT_KIT/ root@EPIC05:/root/A0_P10/
scp -r _moonshot/ root@EPIC05:/root/A0_P10/      # for Stage-0 kernel byte-identity
ssh root@EPIC05
cd /root/A0_P10/A0_P10_PREFLIGHT_KIT/a0preflight
```

## 3. Run

One command runs the full preflight (it stops at the first blocker, by design):

```bash
CUDA_VISIBLE_DEVICES=0 /root/miniforge3/envs/gsplat/bin/python a0p10_run.py --out /root/A0_P10/out
```

Expected staged output (each printed as it completes):

```
=== Stage 0: identity ===              -> /root/A0_P10/out/A0_SOURCE_IDENTITY.json
  [OK] core gsplat_cuda.so … 935ea6c0…  MATCH
  [OK] c0_ext_so … d26cccf4…  MATCH
  [OK] HigsNativeBackward.cu byte-identity … BYTE-IDENTICAL
  [OK] RasterizeToPixels3DGSDevice.cuh byte-identity … BYTE-IDENTICAL
  [OK] checkpoint room/25000 … b31e5a6b…  MATCH
  [OK] frozen camera index 19 img_name == DSCF4819 … MATCH
  PASS=True
=== Stage 2: native graph + raw-frame adjoint (preflight sanity) ===
  raw (1080, 1920, 3) loss=0.04… |v_raw|=…
=== Stage 3: native tile x batch bucket census (bucket_width=128) ===
  tiles=8580 native_buckets=… nonempty=… total_accepted_hits=…
  buckets -> /root/A0_P10/out/A0_BUCKETS.json
=== Stage 4: coefficient dump (top-32 native buckets) ===
  wrote 32 bucket npz shards -> /root/A0_P10/out/buckets
=== Stage 2b: preflight sanity ===
  preflight sanity pass=True
=== Stage 5: oracle-SVD / DCT structure + P10 verdict ===
  structure -> /root/A0_P10/out/A0_STRUCTURE_RESULTS.json  P10 verdict=A0P_…
=== A0-P10 PREFLIGHT COMPLETE ===
```

### Run only the identity gate (cheap, no GPU)

```bash
/root/miniforge3/envs/gsplat/bin/python a0p10_identity.py --out /root/A0_P10/out/A0_SOURCE_IDENTITY.json
```

### Re-run the analyzer on already-dumped shards (no GPU, no re-render)

```bash
/root/miniforge3/envs/gsplat/bin/python oracle_analyze.py --dir /root/A0_P10/out/buckets --out /root/A0_P10/out/A0_STRUCTURE_RESULTS.json
```

## 4. Stop discipline (blockers)

The driver exits non-zero and writes `A0_VERDICT.json` when a frozen gate fails:

| exit | verdict code | meaning |
|---|---|---|
| 3 | `A0P_IDENTITY_BLOCKER` | a frozen source/checkpoint is missing, its sha256 mismatches, or the kernel byte-identity / cam-19 / room-25K sha asserts fail |
| 6 | `A0P_CAMERA_BLOCKER` | frozen camera index 19 `img_name != DSCF4819` on the live GTDataset |
| 4 | `A0P_STATE_BLOCKER` | `densification_info` is missing conics/means2d/colors_eval/opacities/tile state |
| 5 | `A0P_SANITY_BLOCKER` | preflight sanity failed: loss not reproducible, native grads not finite, walk not deterministic, or coeffs not finite / `m`>bucket width |
| 0 | (verdict `A0P_*`) | all gates passed; structure + P10 verdict written |

On any blocker: **stop**, return the artifacts to the research lead, do not
continue, and do not change any frozen threshold/rank/basis to "fix" it.

## 5. The frozen mechanical P10 verdict (oracle SVD only)

The P10 verdict is a **frozen mechanical rule** computed from the **oracle-SVD**
r99 / r995 medians over the top-32 native buckets (DCT is diagnostic and does
**not** affect the verdict):

| verdict | rule |
|---|---|
| `A0P_DROP` | median r99 > 16 for `q_alpha` **OR** joint-`q_color` |
| `A0P_STRONG_PROMOTE` | median r995 ≤ 8 for **BOTH** `q_alpha` and joint-`q_color` |
| `A0P_INCONCLUSIVE` | otherwise |

This is a **preliminary single-cell preflight** signal. The full A0-v2 verdict
(`A0_STRONG/MODERATE_ADJOINT_TENSOR_SIGNAL`, `A0_STRUCTURE_EXISTS_ENCODING_UNSOLVED`,
`A0_NO_USEFUL_ADJOINT_TENSOR_STRUCTURE`, `A0_ALGEBRA_BLOCKER`, `A0_PARITY_BLOCKER`)
requires the formal A0 campaign (all 4 cells + reconstructed-VJP + cost gates) and
is **not** emitted here.

## 6. Outputs

All under `/root/A0_P10/out/`:

- `A0_SOURCE_IDENTITY.json` — Stage 0 (source hashes, kernel byte-identity, checkpoint, cam, env, recipe, frozen constants)
- `A0_PREFLIGHT_SANITY.json` — Stage 2/2b preflight-sanity gate (NOT full A0-v2 parity)
- `A0_BUCKETS.json` — Stage 3 frozen top-32 native buckets (tile/batch id, origin, accepted hits, batch metadata)
- `buckets/bucket_*.npz` — Stage 4 exact FP32 `q_alpha [256,m]`, `q_color [256,m,3]` (primary, real 3-channel), `w_color [256,m]` (secondary), `accepted_mask [256,m]`, `col_gaussians [m]`, per-bucket metadata
- `A0_STRUCTURE_RESULTS.json` — Stage 5 oracle-SVD/DCT per-bucket rank stats + P10 verdict
- `A0_VERDICT.json` — final P10 verdict code + artifact map
- `A0_EXACT_BACKWARD_FACTORIZATION.md` — (static, in the kit) Stage 1 algebra

## 7. Returning artifacts

```bash
cd /root/A0_P10
tar czf A0_P10_RETURN.tar.gz out/
scp root@EPIC05:/root/A0_P10/A0_P10_RETURN.tar.gz <local>/
```

Return the tarball + this runbook to the research lead. **STOP** after packaging —
DSH performs the remote run; do not expand to the full campaign from this kit.

## 8. Frozen identities asserted by Stage 0

| item | expected sha256 |
|---|---|
| core gsplat_cuda.so (frozen C0 lineage) | `935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e` |
| c0_ext_so (P12 publication lineage) | `d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11` |
| fused_ssim_cuda.so (P11-locked loss) | `dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb` |
| publication_trainer_p12.py | `675b0267f3a1a6a0fa63fadb37c8536481c1c7a6de37a4b7f6ffa98a449050c5` |
| gaussian_model.py | `68731e375013a6f24aa4d81e5986402b1ac50f51cddf20688e358be705b1a60b` |
| config.py | `ea8e870476a3ba71c0ae9e1123a6fa5b1cf696bad983bdb6535d6fd04b7acc56` |
| colmap_reader.py | `b30a2a345f7f37cdf3087298750d714d7117e1e93ad742e766075be304729b1b` |
| phase7 dataset.py | `e6d8bfb1034bee102abc176060a8c7783951d253426ea7b5ea27daa16c258402` |
| C0 gaussian_inference.py | `d599aae3c1864993934d65646bfcb33e07e6046e389f028230752200df0c3a8a` |
| checkpoint room/25000 | `b31e5a6b0e733911b736bbad2e4f9d4d804d5929655f64efb51a9e5132fc047d` |
| checkpoint room/20000 (reference) | `c4e41cfef79e50980406fc1fa148af8d04bc8b37b801c36da5cddb2ce8058718` |
| HigsNativeBackward.cu | byte-identity vs `_moonshot/HigsNativeBackward.cu` (sha assert) |
| RasterizeToPixels3DGSDevice.cuh | byte-identity vs `_moonshot/RasterizeToPixels3DGSDevice.cuh` (sha assert) |
| frozen camera index 19 | `img_name == DSCF4819` |

C0 env (set by `bootstrap_c0`, recorded in the identity JSON):
`HIGS_PX_RUNTIME=2`, `HIGS_BWD_SCALAR_ADJOINT=scalar_adjoint`,
`HIGS_BWD_H8_MR=1`, `HIGS_BWD_ABSGRAD=1`, `HIGS_DISABLE_F9` unset (F9 ON).
Recipe: `seed=42`, `loss=fused (0.8*L1 + 0.2*(1-SSIM_fused[padding=same]))`,
`eps2d=0.3`, `sh_degree=3`, `tile_size=16`, `packed=False`, `radius_clip=0`,
`render_mode=RGB`, `background=black`, `absgrad=True`, 1080p.

Frozen constants (authoritative `Common.h`): `ALPHA_THRESHOLD=1/255`,
`MAX_ALPHA=0.99`, `TRANSMITTANCE_THRESHOLD=1e-4`, `MIN_ONE_MINUS_ALPHA=1e-6`.

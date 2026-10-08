# P12 frozen C0 source certification

Date: 2026-09-26
Role: DSH-2 (P12 audit / preflight support lane)
Evidence artifacts: `p12_support_audit/certification/` (local) = `/root/epic05_p12_support/certification/` (EPIC-05, byte-identical)

## HISTORICAL P8 C0

- **Repository root:** `/mnt/storage_pool/liaoyuanjun/higs_c0_final30k_worktree` (mx / bms-39468022-001, user liaoyuanjun)
- **Base commit:** `77ab983ffe43420b2131669cb35776b883ca4c3c` ("[NV] uint64 pixel mask, sparse-bwd robustness fixes, CUDA 12.8 build, ortho camera (#1023)", 2026-07-09)
- **Dirty status:** DIRTY by construction — 25 porcelain entries; the HiGS experimental renderer sources are staged/uncommitted on the base commit. Renderer identity is pinned by the source manifest, not by any commit.
- **Renderer source manifest SHA256:** `4baf54c0f7e220436f6d54ed8488fb4e671ca757f5710fced41cad5886543170` (574 normative files, lexicographically canonical, stored outside any renderer checkout)
- **Renderer tree SHA256:** `b483a2c0e40d4434a1471354419bde0e453c27ade9ac1a121dc135b8b143c98a` (full non-transient index, 2174 files)
- **C0 .so SHA256:** ext `9baf8655f859f3e0f456e99a2d8e5b3ddfe414fe17a1a3e9d8072e0afed95b91` (built 2026-09-23 12:03:14 +0800) + core `361b216bcc11609a0ebb8fb44ad2e0c6170948112b6294e85123df45541c8c98`; used identically by all 13 c0 final30k runs and all 26 a0/a1 Stage-A runs
- **Submodule identities:** glm `2d4c4b4dd31fde06cfffad7915c2b3006402322f` (declared in .gitmodules; content present in-tree, 279 headers pinned file-by-file in the manifest); googletest `d72f9c8aea6817cdf1ca0ac10887f328de7f3da2` (declared; not populated; not a renderer build input)
- **Build metadata:** torch 2.9.1+cu128, nvcc Build cuda_12.8.r12.8/compiler.35583870_0 (12.8.93), CUDA_HOME `/mnt/storage_pool/liaoyuanjun/higs-13scene-env`, A100-PCIE-40GB (sm_80); full cflags/cuda_cflags/ldflags/source list recorded in `P8_C0_SOURCE_IDENTITY.json` build_metadata
- **Patch identity:** stage-1 `absgrad_compat_patch.py` (sha `5f123f80…`, 21 edits) then stage-2 `fix_bg_kernel_arg.py` (sha `7b506d48…`); timeline 11:44:47 → 12:02:47 → .so 12:03:14 proves the certified .so was compiled from the post-fix tree; `unexplained_renderer_differences: []`

## P12 C0

- **Planned source:** frozen copy `/root/epic05_frozen/higs_c0_final30k_worktree`; canonical clean snapshot repo `/root/epic05_p12_support/P12_C0_RENDERER_REPO` @ `2c6d9685b0eca852f6749ff4d64dc5a3ef942237` (git tree `7fef029e…`, branch main, 2174 tracked files, clean with --untracked-files=all; deterministic pinned-author commit recipe)
- **Renderer source manifest SHA256:** `4baf54c0f7e220436f6d54ed8488fb4e671ca757f5710fced41cad5886543170`
- **Renderer tree SHA256:** `b483a2c0e40d4434a1471354419bde0e453c27ade9ac1a121dc135b8b143c98a`
- **C0 .so SHA256:** ext `9baf8655f859f3e0f456e99a2d8e5b3ddfe414fe17a1a3e9d8072e0afed95b91` + core `361b216bcc11609a0ebb8fb44ad2e0c6170948112b6294e85123df45541c8c98` (prebuilt historical pair, reused without rebuild; EPIC copies at `/root/epic05_frozen/mx_prebuilt_so/c0_ext_mx.so` and `gsplat_cuda_mx.so` — both re-hashed on EPIC, values match)

## COMPARISON

- **Renderer source identical:** YES — `P12_C0_VS_P8_C0_RENDERER_DIFF.json` (sha `3f6fca6a…`) reports status `IDENTICAL_RENDERER_SOURCE`: normative 574/574 identical (0 missing, 0 extra, 0 hash mismatch); full index 2174/2174 identical; both tree digests equal `b483a2c0…`.
- **Loss-plumbing-only difference:** YES — the only P12-intended changes live outside the renderer tree: `publication_trainer_p12.py` (sha `689ec892…`), `p12_fused_loss.py` (sha `a6087288…`), `P12_TRAINER_PATCH.json` (sha `493840b1…`). None is part of the certified manifest or tree digest. Common fused loss 0.8*L1 + 0.2*(1−SSIM), padding=same, applied identically to all four arms; loss algorithm unmodified.
- **Build-mechanical differences:** NONE between P8 C0 and P12 C0 (binary reused as-is, no rebuild). Historical-internal note: the stage-2 fix reverted an over-matched 5th launcher argument (`bfbf0230` → `6efb9d25` for HigsNativeBackward.cu) before the certified .so was compiled — fully documented in the identity record.
- **Unexpected differences:** NONE — zero renderer differences of any kind; no algorithmic CUDA/C++ difference exists. The `UNEXPECTED` classification list is empty; no reconciliation or patching was performed.

## A0/A1

- **Same source tree:** YES — both bind C0_WT `/mnt/storage_pool/liaoyuanjun/higs_c0_final30k_worktree` (trainer shas `58e6cc2c…` c0, `8b8e0734…` a0/a1); tree identity = certified manifest `4baf54c0…` / digest `b483a2c0…`.
- **Same binary:** YES — all 39 run records (13 c0 + 13 a0 + 13 a1) carry ext `9baf8655…` + core `361b216b…`; every a0/a1 record explicitly asserts `binary_same_as: c0 (ext 9baf8655 + core 361b216b)`. Live re-verification 2026-09-26: 39/39 records, 0 violations.
- **F9 runtime-only:** YES — the only difference between A0 and A1 execution is the runtime switch `HIGS_DISABLE_F9` (=1 for A0, unset/default-on for A1); identical physical .so.
- **Scalar:** OFF in both (HIGS_BWD_SCALAR_ADJOINT unset in all 26 a0/a1 records). **H8:** OFF in both (HIGS_BWD_H8_MR unset in all 26 records).
- Consequence: the Codex `p12_diff_guard.py` F9 allowlist is legitimately empty (`{}`) — A0/A1 source trees are identical to each other and to P12 C0, and F9 is selected solely by a runtime switch in the same frozen binary. P12-B guard matrices reproduce this exactly.

## CODEX EXPECTED VALUE

- **`expected_renderer_source_sha256` = `4baf54c0f7e220436f6d54ed8488fb4e671ca757f5710fced41cad5886543170`**
- `renderer_source_manifest` = `/root/epic05_p12_support/certification/P8_C0_FROZEN_SOURCE_MANIFEST.json`
- `expected_renderer_commit` = `2c6d9685b0eca852f6749ff4d64dc5a3ef942237`
- `renderer_repo` = `/root/epic05_p12_support/P12_C0_RENDERER_REPO`
- `--f9-allowlist` = `{}` (see A0/A1 section)
- Pre-launch integration items (blocking-for-launch): rebind the P12 trainer C0_SO to `/root/epic05_frozen/mx_prebuilt_so/c0_ext_mx.so` (9baf8655…) + `gsplat_cuda_mx.so` (361b216b…) — it currently points at the EPIC-local rebuild `d26cccf4…`, which is a different binary of the same lineage; run the guard before any renderer import.
- Caveat: the EPIC 3dgs-publication-epic05 campaign's own rebuild (`d26cccf4` / `935ea6c0`) is NOT the frozen baseline; P12's frozen C0 identity is the mx build certified above. Do not mix.

## VERDICT: SOURCE_CERTIFIED

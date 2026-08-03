# Patches

- `higs-differentiable.patch` — the HiGS differentiable training implementation,
  applied to the gsplat source tree (upstream baseline `77ab983`). 17 files:
  native CUDA backward kernels (`HigsNativeBackward.cu/.h`), visible-gather
  kernels (`GatherVisible.cu/.h`), `ext.cpp` bindings, the CUDA 12.9
  `Utils.cpp` event fix, Python autograd layer, and tests.

How it is maintained:
- Authoritative source: `artifacts/renderer-sources/gsplat` (git-ignored working tree).
- Regenerate with `git -C artifacts/renderer-sources/gsplat diff --cached` (LF line endings).
- Verify with `git apply --check patches/higs-differentiable.patch` on a clean `77ab983` checkout.

Full background: `reports/higs-trainability-implementation.md` (Round 30).

Third-party/environment build patches live in `scripts/linux/patches/` and
`third_party_patches/` instead.

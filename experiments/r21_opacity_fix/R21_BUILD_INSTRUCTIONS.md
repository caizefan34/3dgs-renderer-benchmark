# R21 isolated candidate preparation

Status: IMPLEMENTED; local Python checks and extracted-kernel GPU checks PASS;
MX candidate deployment/full parity NOT_EXECUTED by Codex.

This is a Python-only overlay. **Do not rebuild or replace either .so.** The
existing R20 extension already has True AbsGrad and the frozen main extension
already supports opacity-aware projection. Use the same MX environment:
`/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python3`.

Copy/check out this branch's candidate delivery and its pinned R20/R21 test
dependencies in the MX repository. The manifest contains the implementation
commit and exact payload SHA256; dependency source hashes normalize CRLF only.
The metadata commit follows the implementation commit to avoid a self-referential
commit/hash requirement. The manifest excludes itself and the separate
`R21_MANIFEST.sha256` checksum file. The latter authenticates the manifest bytes.

```bash
set -euo pipefail
export R21_REPO=/home/liaoyuanjun/3dgs-renderer-benchmark
export R21_TASK="$R21_REPO/experiments/r21_opacity_fix"
export R21_PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python3
export R21_STD=/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx
export R21_R20=/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_ABSGRAD_PARITY_R20_MX/gsplat-higs-r20
export R21_CANDIDATE=/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_OPACITY_FIX_R21_P4_MX/gsplat-higs-r21-opacity
export R21_OUT=/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_OPACITY_FIX_R21_P4_MX/results
cd "$R21_TASK"
sha256sum -c R21_MANIFEST.sha256
"$R21_PY" R21_PREPARE_CANDIDATE.py --verify-delivery
test ! -e "$R21_CANDIDATE"
test ! -e "$R21_OUT"
mkdir -p "$R21_OUT/logs"
"$R21_PY" R21_PREPARE_CANDIDATE.py --r20-root "$R21_R20" \
  --candidate-root "$R21_CANDIDATE" > "$R21_OUT/logs/prepare.json"
"$R21_PY" R21_PREPARE_CANDIDATE.py --verify --candidate-root "$R21_CANDIDATE"
"$R21_PY" R21_VISIBILITY_REGRESSION_TESTS.py --cpu --output "$R21_OUT/local_contracts.json"
```

Preparation validates the original R20 wrapper, all four R20 overlay sources,
frozen projection/rendering/AccuTile dependencies, and these exact binaries:

| Binary | SHA256 |
|---|---|
| Frozen main `gsplat/csrc.so` | `40a009837e75620454059cfdbb5088cc08423f281e4c734da3053b727d39df50` |
| Validated R20 HiGS extension | `b3c3e2e037f5c7f8921bc8b1e62c6bf924de88e23380a4bd195343c1146142ac` |

It copies the tree to a new path, overlays one wrapper, verifies every other
copied file against R20, and writes `R21_BUILD_IDENTITY.json`. No hardlinks,
in-place edits, package install, JIT build or checkpoint writes are involved.
The existing `R20_BUILD_IDENTITY.json` is preserved because its binary remains
identical. A mismatch aborts; export and audit changed sources instead of
disabling checks. Failed preparation leaves a partial new directory for
inspection; use another fresh directory when retrying.

P4 workers and history capture preload the two compiled modules by their exact
paths, before package initialization, and check the loaded paths and SHA256.
Import/ABI failures abort. They cannot silently use the backend loader's JIT
fallback. The two added CPU tests cover wrong loaded paths/bytes and import
failure; the CPU suite now has 15 tests. Production wrapper bytes are unchanged
by this additional validation.

Public API is unchanged. Internal projection calls now require activated
opacity `[1,N]` / `[1,Nvisible]`. `eps2d=.1` remains the explicit R20 trainer
setting. Neither thresholds nor sigmoid/exp/quaternion activation is changed.
The R21 capture adapter only accepts this pinned candidate; the original R21
capture tool still correctly rejects it as a non-R20 wrapper.

Local reproduction on the development Windows machine:

```powershell
python experiments/r21_opacity_fix/R21_VISIBILITY_REGRESSION_TESTS.py --cpu --output "$env:TEMP/r21_cpu.json"
cmd /c experiments\r21_opacity_fix\_local_projection_check.cmd
```

The second command compiles an unmodified extracted frozen projection kernel
using frozen Utils/Common headers, local PyTorch/CUDA/MSVC and local GLM.
Its result is `local_checks/PROJECTION_RESULT.json`: 20 scoped cases passed,
including exact radii/visibility versus recorded MX data for 80 real rows.
It does not execute frozen Linux binaries, raster tiles, backward or training.
GLM is a local build dependency, not the evidence used to infer projection
semantics. Preserve the recorded build command and dependency hashes.

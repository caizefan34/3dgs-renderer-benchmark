# Renderer environment setup report

> Update: the original base-environment blocker documented below has been resolved by a separate `original3dgs` environment. See `original-fix-report.md`. The base environment itself remains unchanged.

- Timestamp: `2026-07-17T09:35:00+08:00`
- Scope: `base`, `gsplat`, and `tcgs` Conda environments plus immutable renderer sources.
- Constraints observed: benchmark rules, protocol, suite, and renderer adapter source files were not changed by this setup task; existing CUDA binaries were preferred over recompilation; original and TC-GS extensions were kept in separate environments.

## Commands executed

Environment inventory and verification:

```powershell
conda env list
<env-python> -c "import torch, importlib.metadata ..."
<env-python> -m pip check
<env-python> src/run_benchmark.py --list-renderers
```

Dependency installation was launched for each environment with:

```powershell
<env-python> -m pip install --disable-pip-version-check `
  -r requirements-benchmark.txt -r requirements-quality.txt pytest
```

The `gsplat` and `tcgs` commands completed. The parallel `base` pip child stopped making progress while connected to PyPI and was terminated after approximately 15 minutes. Its missing packages were then installed without changing torch:

```powershell
C:\Users\36570\miniconda3\python.exe -m pip install `
  --disable-pip-version-check --only-binary=:all: `
  psutil nvidia-ml-py pytest
```

Renderer sources were cloned with the command pattern recorded in `source-clone-log.md`. The Speedy rasterizer submodule was then materialized without building it:

```powershell
git -C artifacts/renderer-sources/speedy-splat submodule update `
  --init --depth 1 submodules/diff-gaussian-rasterization
```

## Final environment versions

| Environment | Python | PyTorch | Torch CUDA | numpy | Pillow | LPIPS | psutil | nvidia-ml-py | remotezip | google-crc32c | pytest |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base` | 3.13.13 | 2.13.0+cu130 | 13.0 | 2.5.1 | 12.3.0 | 0.1.4 | 7.2.2 | 13.610.43 | 0.12.3 | 1.7.1 | 9.1.1 |
| `gsplat` | 3.10.20 | 2.12.1+cu130 | 13.0 | 2.2.6 | 12.3.0 | 0.1.4 | 7.2.2 | 13.610.43 | 0.12.3 | 1.8.0 | 9.1.1 |
| `tcgs` | 3.10.20 | 2.12.1+cu130 | 13.0 | 2.2.6 | 12.3.0 | 0.1.4 | 7.2.2 | 13.610.43 | 0.12.3 | 1.8.0 | 9.1.1 |
| `original3dgs` | 3.10.20 | 2.12.1+cu130 | 13.0 | 2.2.6 | 12.3.0 | 0.1.4 | 7.2.2 | 13.610.43 | 0.12.3 | 1.8.0 | 9.1.1 |

PyTorch was unchanged in all three environments. `pip check` passes in `gsplat` and `tcgs`. `base` retains two pre-existing conflicts: torchvision `0.27.1+cu130` declares torch `2.12.1`, while base has torch `2.13.0+cu130`; viser `1.0.30` requires rich `<15`, while base has rich `15.0.0`.

## Immutable source verification

| Family | Source path | HEAD |
|---|---|---|
| original 3DGS | `artifacts/renderer-sources/original-diff-gaussian-rasterization` | `9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0` |
| gsplat / HiGS | `artifacts/renderer-sources/gsplat` | `77ab983ffe43420b2131669cb35776b883ca4c3c` |
| Speedy-Splat | `artifacts/renderer-sources/speedy-splat` | `34c45c6d9b8bd6110231864f2f358b6d3abbf73d` |
| TC-GS | `artifacts/renderer-sources/3DGSTensorCore` | `0bb82f88fde211c34b42e1497f0fc7265461592b` |

All four checkouts are detached and clean. Full clone commands and remote verification are in `source-clone-log.md`.

## Runtime source and binary binding

No CUDA extension was rebuilt.

### gsplat environment

`zzz_benchmark_renderer_sources.pth` imports `artifacts/environment-setup/benchmark_renderer_preload.py`. The helper:

1. puts the pinned gsplat and Speedy source checkouts on `sys.path`;
2. loads the previously verified CUDA 13 / Python 3.10 cached binaries;
3. binds them to the module names expected by the pinned gsplat source.

Copied build parameter records:

- `gsplat_cuda_build_params.json`
- `gsplat_scene_cuda_build_params.json`
- `gsplat_higs_inference_build_params.json`

Runtime binary SHA-256 values:

| Binary | SHA-256 |
|---|---|
| `gsplat_cuda.pyd` | `5345ac75a8b9cfcafd2e32b182cd12725b51f9ea8ab9eedf4baff47fb093a951` |
| `gsplat_scene_cuda.pyd` | `5f27c8e180105d94b7358ae53047fd522e5f194490ab5c078b9db597e923ebd7` |
| HiGS inference `.pyd` | `89430e5bfca26b4ad8cfa4fd4101e9f02b68e0c5407606b500e35afb89029c45` |
| Speedy `_C.cp310-win_amd64.pyd` | `0b921888439a590e9afbc89df448725cdd4c86a3f38ba0d898cbc4db6bb83b62` |

Fresh-process adapter metadata verification:

| Renderer | Available | Runtime version | Reported commit |
|---|---|---|---|
| `gsplat` | yes | 1.5.3 | `77ab983ffe43420b2131669cb35776b883ca4c3c` |
| `gsplat_higs` | yes | 1.5.3 | `77ab983ffe43420b2131669cb35776b883ca4c3c` |
| `speedy_splat` | yes | unknown | `34c45c6d9b8bd6110231864f2f358b6d3abbf73d` |

The installed gsplat distribution metadata remains the old `1.4.0+pt21cu118` wheel, but the imported runtime module is the pinned Git source at version `1.5.3`; renderer metadata therefore reports the pinned source version and commit. The cached CUDA binaries are the earlier CUDA 13 build whose build-parameter files are preserved above.

### TC-GS environment

The installed TC-GS Python wrapper exactly matched the wrapper in the pinned top-level repository. Its existing `.pyd` was copied into the pinned checkout and loaded through a source-path `.pth` entry. This keeps the extension isolated from base original 3DGS.

- TC-GS `_C.cp310-win_amd64.pyd` SHA-256: `f9ee6aa822fa59a2a4c13c5b82956329df9fce18e175162519924bbd1926aad9`
- Fresh-process status: available
- Reported commit: `0bb82f88fde211c34b42e1497f0fc7265461592b`

### Base original 3DGS blocker

The existing base extension is importable, but it cannot be made an honest Tier A installation of the registry commit without changing benchmark code or rebuilding/patching:

- installed wrapper SHA-256 differs from the pinned checkout;
- installed wrapper returns four values (`color`, `radii`, `depth`, `alpha`) and has no discoverable VCS commit;
- pinned commit `9c5c...` returns three values (`color`, `radii`, `invdepths`);
- the benchmark adapter currently unpacks four values.

Relabeling the existing binary as `9c5c...` would create false provenance, while compiling the exact pinned commit would still hit the adapter return-value mismatch. Per instruction, no rebuild was attempted. Current metadata remains:

```json
{
  "available": true,
  "version": "0.0.0",
  "commit_hash": null
}
```

## Safe execution scope

Do not run `benchmark run all` from a single environment. Use renderer-specific interpreters so the same-named original and TC-GS extensions never overwrite or masquerade as each other:

```powershell
C:\Users\36570\miniconda3\envs\original3dgs\python.exe benchmark.py run original_3dgs

C:\Users\36570\miniconda3\envs\gsplat\python.exe benchmark.py run gsplat
C:\Users\36570\miniconda3\envs\gsplat\python.exe benchmark.py run gsplat_higs
C:\Users\36570\miniconda3\envs\gsplat\python.exe benchmark.py run speedy_splat

C:\Users\36570\miniconda3\envs\tcgs\python.exe benchmark.py run tcgs
```

`src/run_benchmark.py --list-renderers` confirms the requested renderers are importable in the corresponding environments. No full scene render was attempted because canonical Matrix assets are not yet staged.

# Tier A original_3dgs environment report

- Date: `2026-07-17`
- Environment: `C:\Users\36570\miniconda3\envs\original3dgs`
- Renderer source: `artifacts/renderer-sources/original-diff-gaussian-rasterization`
- Required renderer commit: `9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0`
- Result: PASS — importable, CUDA smoke-tested, and exact commit discoverable by the benchmark adapter.

## Benchmark code change

Only the original diff-gaussian adapter and its focused unit test were changed:

- `src/renderers/diff_gaussian_renderer.py`
- `tests/test_renderers.py`

The adapter now:

1. supplies `antialiasing=False` only when the installed settings type supports that parameter;
2. accepts the pinned upstream three-value return and existing four-value forks;
3. still uses only the first returned value as the RGB render.

No protocol, suite, ranking, workload, resolution, trajectory, or quality code was changed by this task.

### Test-first evidence

The three-value test was added before the implementation.

Initial result:

```text
FAILED tests/test_renderers.py::Original3DGSRendererTest::test_accepts_pinned_three_value_result
ValueError: not enough values to unpack (expected 4, got 3)
```

After adding the return compatibility, the pinned settings signature exposed a second failure:

```text
TypeError: Settings.__init__() missing 1 required positional argument: 'antialiasing'
```

After the conditional settings argument was implemented:

```text
2 passed in 4.31s
```

The pre-existing four-value test remains part of `Original3DGSRendererTest`, so both interfaces are exercised.

## Environment creation and isolation

The known-good Python 3.10/cu130 environment was cloned to avoid downloading or changing PyTorch:

```powershell
conda create --clone gsplat -n original3dgs -y
```

Renderer contamination was removed from the clone:

```powershell
python -m pip uninstall -y gsplat fast-gauss
```

The cloned gsplat preload `.pth`, legacy gsplat `.pth`, and Speedy module path were disabled. Stale namespace directories for gsplat and diff-gaussian were moved aside before installing original 3DGS. The resulting renderer availability is:

```text
Available: ['diff_gaussian', 'original_3dgs']
```

Final environment:

| Item | Version |
|---|---|
| Python | `3.10.20` |
| PyTorch | `2.12.1+cu130` |
| PyTorch CUDA runtime | `13.0` |
| GPU | `NVIDIA GeForce RTX 5070 Laptop GPU` |
| Compute capability | `12.0` |
| numpy | `2.2.6` |
| Pillow | `12.3.0` |
| lpips | `0.1.4` |
| psutil | `7.2.2` |
| nvidia-ml-py | `13.610.43` |
| remotezip | `0.12.3` |
| google-crc32c | `1.8.0` |
| pytest | `9.1.1` |

`pip check` result: `No broken requirements found.`

## Source and build provenance

Top-level checkout:

```text
HEAD 9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0
detached HEAD
worktree clean
```

Pinned submodule:

```text
third_party/glm 5c46b9c07008ae65cb81ab79cd677ecc1934b903
```

The original partial clone could not be re-cloned through `git+file` because pip requested a promisor object not present locally. No package was installed by that attempt. The successful installation therefore used an editable build directly from the clean checkout:

```powershell
python -m pip install -v --no-build-isolation --no-deps --force-reinstall `
  -e C:\Users\36570\Documents\Codex\3dgs-renderer-benchmark\artifacts\renderer-sources\original-diff-gaussian-rasterization
```

Build environment:

```text
Visual Studio 2022 Community 17.14.15 / MSVC 19.44.35217
CUDA toolkit 13.3 / nvcc 13.3.73
TORCH_CUDA_ARCH_LIST=12.0
MAX_JOBS=4
CL=/Zc:preprocessor
CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
```

The first valid compiler invocation failed only because GLM had not been initialized. After checking out the parent-pinned GLM commit, the next build completed successfully and produced:

```text
diff_gaussian_rasterization-0.0.0-0.editable-cp310-cp310-win_amd64.whl
```

Built extension:

- Path: `artifacts/renderer-sources/original-diff-gaussian-rasterization/diff_gaussian_rasterization/_C.cp310-win_amd64.pyd`
- SHA-256: `f53bb6191e8b5cd85fa5bea46a0d49095aef3b9176b9d0dc19e552e236b2befc`
- CUDA code generation: `compute_120, sm_120`

The generated `.pyd` and Python cache are locally excluded through `.git/info/exclude`; no tracked upstream file was changed and `git status` is clean.

PEP 610 records the real editable source directory:

```json
{
  "dir_info": {"editable": true},
  "url": "file:///C:/Users/36570/Documents/Codex/3dgs-renderer-benchmark/artifacts/renderer-sources/original-diff-gaussian-rasterization"
}
```

The benchmark metadata resolver then follows the imported module path to that Git checkout. Verified adapter metadata:

```json
{
  "implementation": "graphdeco-inria/diff-gaussian-rasterization",
  "version": "0.0.0",
  "source_url": "https://github.com/graphdeco-inria/diff-gaussian-rasterization",
  "commit_hash": "9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0"
}
```

No commit metadata was manually inserted or fabricated.

## Runtime checks

Availability/import:

```text
original_3dgs available: true
module file: artifacts/renderer-sources/original-diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py
```

CUDA adapter smoke used four Gaussians and a generated 64×36 camera:

```json
{
  "shape": [36, 64, 3],
  "dtype": "torch.float32",
  "device": "cuda:0",
  "finite": true,
  "min": 0.0,
  "max": 0.4984983503818512,
  "sum": 42.39805603027344
}
```

This confirms the pinned three-value CUDA path executes through the benchmark adapter on `sm_120`.

## Tests

Focused adapter tests:

```text
2 passed
```

Repository test directory:

```text
98 passed, 1 skipped in 7.23s
```

Running bare `pytest -q` also discovers upstream tests inside `artifacts/renderer-sources/gsplat`; that unrelated collection attempts to JIT-build gsplat and fails without an activated compiler shell. The authoritative repository test command is therefore `python -m pytest tests -q`, which passed as shown above.

## Tier A invocation

Use this interpreter specifically for original 3DGS; do not use the TC-GS or gsplat environments:

```powershell
C:\Users\36570\miniconda3\envs\original3dgs\python.exe benchmark.py run original_3dgs
```

Canonical assets must be staged before that command can produce Tier A records.

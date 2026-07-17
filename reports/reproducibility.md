# Reproducibility and resume instructions

## Verify the repository and assets

```powershell
git rev-parse HEAD
Get-FileHash -Algorithm SHA256 benchmark\protocol.json
```

Expected base commit: `b46e8f27fbc3beea89a12f25c35ce8b296f24cd9`.

Expected protocol SHA-256: `892e18890501c408dc6746af69f17e16973604f0c02a3caddf1954d3bf1fede2`.

Use the Git commit containing these reports; it includes the evidence-export and adapter compatibility changes required by this run.

## Environment

Use Python 3.10.20, PyTorch 2.12.1+cu130, CUDA toolkit 13.3, and the pinned renderer checkouts listed in `renderer_report.md`. The current Windows build details are in `../artifacts/environment-setup/`.

The per-process backend selector used for the balanced all-renderer run is:

`../artifacts/environment-setup/sitecustomize.py`

## Run on an NVML-capable host

First confirm that a live CUDA process has a numeric `used_gpu_memory` value:

```text
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
```

Do not continue if the field is `N/A`.

Then run the fixed suite:

```powershell
$env:CUDA_VISIBLE_DEVICES='0'
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH="$PWD\artifacts\environment-setup"
C:\Users\36570\miniconda3\envs\gsplat\python.exe benchmark.py run all
C:\Users\36570\miniconda3\envs\gsplat\python.exe benchmark.py report --output-dir reports\generated\ranking
```

The run must produce valid `metrics.json` files for all five required cases for both `original_3dgs` and at least one candidate. Do not copy the partial Train numbers into `metrics.json` or substitute framework memory.

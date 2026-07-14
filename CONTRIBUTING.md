# Contributing

Thanks for helping make 3DGS renderer comparisons more reproducible. Small
fixes, new adapters, additional hardware results, and protocol improvements are
all welcome.

## Development setup

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

Renderer packages are optional because most are CUDA extensions with
environment-specific build requirements. Install only the backends needed for
your contribution.

## Good first contributions

- reproduce the protocol on another NVIDIA GPU and submit the full environment
  metadata with the generated JSON report;
- publish a legally redistributable trained scene with a stable URL and
  checksum, then move its manifest entry from `planned` to `available`;
- add a CPU-safe contract test for an existing renderer adapter;
- improve Linux or Windows build notes with an exact, verified toolchain.

Comment on or open an issue before taking on a new renderer so work is not
duplicated.

## Adding a renderer

1. Implement `RendererAdapter` in `src/renderers/` and register it in
   `src/renderers/__init__.py`.
2. Keep scene activation and static packing in `prepare_scene`; do not hide
   per-frame work outside the timed render call.
3. Add a CPU-safe adapter contract test using a mocked backend.
4. Compare output against a designated reference renderer with PSNR and SSIM.
5. Record the upstream repository, exact commit, runtime version, and patches.
6. Report GPU-event and end-to-end latency separately.

A renderer must pass finite-output, camera-change, and quality checks before
its speed result is described as verified.

## Submitting benchmark results

Include all of the following in the pull request:

- GPU model and VRAM;
- operating system and driver version;
- Python, PyTorch, CUDA runtime, and CUDA toolkit versions;
- renderer repository and exact commit;
- scene source and Gaussian count;
- resolution, camera preset, warmup, frames, and repeats;
- mean, median, P95, P99, peak VRAM, PSNR, and SSIM;
- whether GPU clocks were locked.

Do not compare numbers produced from different scenes, cameras, resolutions,
quality thresholds, or timing boundaries as if they were a leaderboard.

## Pull requests

Keep changes focused. Run the CPU tests and compile check before opening a pull
request, and explain any test that cannot run in your environment. Bug reports
and renderer proposals can start from the repository issue templates.

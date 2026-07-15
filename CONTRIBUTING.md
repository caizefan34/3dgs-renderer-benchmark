# Contributing

Contributions that improve reproducibility, backend coverage, protocol
clarity, or hardware coverage are welcome. Keep each pull request focused on
one verifiable outcome and preserve existing benchmark artifacts.

## Development Setup

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

Renderer packages are optional CUDA extensions. Install only the backend
needed for the contribution, and import it lazily so CPU-only test collection
continues to work.

## Contribution Workflow

1. Open an issue for a new renderer or protocol change to avoid duplicated
   work.
2. Create a small branch and add a failing test that expresses the desired
   behavior.
3. Implement the smallest change that passes the test.
4. Run the full CPU suite and any relevant opt-in GPU regression.
5. Update protocol or onboarding documentation when public behavior changes.
6. Include generated JSON and environment metadata for benchmark submissions.

Do not reformat or refactor unrelated modules. Do not edit generated
leaderboard values by hand.

## Adding a Renderer

New adapters implement the strict `RendererAdapter` ABC in
`src/adapters/base.py`. Follow the complete
[How to add a new renderer](docs/adding-a-renderer.md) guide for a copy-paste
template, registration steps, output requirements, and regression checklist.

A renderer is eligible for a quality-gated result only when it:

- returns finite float32 RGB, depth, and alpha tensors at the requested
  resolution;
- renders at least two distinct camera poses correctly;
- passes the declared PSNR, SSIM, and LPIPS thresholds;
- reports raw ordered latency samples and peak allocated GPU memory;
- differs by less than 0.01 dB across repeated deterministic renders;
- records upstream source, commit, build environment, and local patches.

Keep static packing in checkpoint loading and all camera-dependent work in the
timed render call. Report CUDA-event and end-to-end latency separately.

## Tests

The default suite must remain CPU-safe:

```bash
python -m unittest discover -s tests -v
```

Run the cross-backend reproducibility test only on a configured CUDA host:

```bash
RUN_RENDERER_REGRESSION=1 python -m unittest \
  tests.test_adapters.RendererReproducibilityRegressionTest -v
```

On PowerShell:

```powershell
$env:RUN_RENDERER_REGRESSION = "1"
python -m unittest tests.test_adapters.RendererReproducibilityRegressionTest -v
```

Tests must release renderer references and call `torch.cuda.empty_cache()`
between GPU cases. A test may skip an unavailable optional CUDA backend, but
its reason must identify the missing package or platform capability.

## Submitting Benchmark Results

Include all of the following in the pull request:

- GPU model, VRAM, driver, and clock-lock status;
- operating system, Python, PyTorch, CUDA runtime, and toolkit versions;
- renderer repository, exact commit, build command, and patch files;
- scene source, checkpoint hash, Gaussian count, and reference-image hashes;
- camera manifest hash, resolution, warmup, frames, and repeats;
- raw frame times, mean, median, P95, P99, FPS, and peak memory;
- PSNR, SSIM, LPIPS, thresholds, and quality-gate result.

Results from different scenes, trajectories, resolutions, timing boundaries,
or quality references belong to separate cohorts. Missing metrics remain
`null`; never estimate them.

## Documentation and Style

Use clear academic English, one sentence per line when practical, fenced code
blocks with a language identifier, and descriptive link text. Keep headings
hierarchical and tables aligned. Placeholder benchmark data must use the form
`> **Note:** ...` and must state how to generate the missing measurement.

## Pull Request Checklist

- [ ] The change is scoped to the stated issue.
- [ ] New behavior has a test.
- [ ] CPU tests and compile checks pass.
- [ ] Relevant GPU tests pass or have an explained skip.
- [ ] Public behavior and protocol changes are documented.
- [ ] Benchmark claims include raw artifacts and provenance.
- [ ] No missing measurement has been inferred or fabricated.

By contributing, you agree that your changes are licensed under the
repository's MIT License.

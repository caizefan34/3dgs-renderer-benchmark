# Evaluation Formulas

The current formulas are specified in [methodology.md](methodology.md) and implemented in `src/benchmark_matrix.py`.

Primary outputs are raw FPS/frame time, VRAM, startup/load metrics, PSNR, SSIM, and LPIPS.
The efficiency score divides a bounded three-metric quality utility by frame-time × VRAM resource cost.
It is a versioned decision aid, not a substitute for constraint-based or Pareto comparison.

Pre-Matrix quality-adjusted FPS formulas are legacy diagnostics and must not appear in v3 rankings.

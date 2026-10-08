# MomentGrad-GS — compressed adjoint integration for Gaussian Splatting backprop

Independent research lane. Tests whether a Gaussian's per-pixel backward field over its
screen footprint can be compressed into K quadrature/moment samples instead of summing over
all covered pixels.

Status: M1 gate FAILED -> DROP_GRADIENT_FIDELITY (see out/VERDICT.md, out/M1_REPORT.md).
Hardware: RTX 5070 Laptop (not A100). gsplat 1.4 CUDA backend cannot JIT-compile here
(no MSVC), so a self-contained pure-PyTorch oracle rasterizer is used. M6/M7 (CUDA/A100)
are BLOCKED on this host.

## Layout
- scene.py            COLMAP (MipNeRF 360) loader + Gaussian init
- oracle.py           pure-torch 3DGS rasterizer (per-Gaussian alpha/T/adjoint moments)
- m1_quadrature.py    K-point Gauss-Hermite quadrature vs canonical footprint integral
- m1_run.py           M1 gate runner (room/garden/bicycle)
- m1_smooth_control.py  smooth-field control (isolates field non-smoothness)
- out/                M1_REPORT.md, VERDICT.md, m1_quadrature.json, M1_SMOOTH_CONTROL.json

## Run
    python momentgrad_gs/m1_run.py
    python momentgrad_gs/m1_smooth_control.py

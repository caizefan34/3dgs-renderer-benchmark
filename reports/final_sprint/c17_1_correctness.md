# C17-1 correctness status

Historical representation evidence is strong: 34,800,973 intersections and
25,932 tiles had zero missing, extra, duplicate, depth-order violations, and
100% tile-order match. It is not a substitute for this CUDA implementation's
gate.

This implementation was not loadable on the A100 host: its shared gsplat 1.5.3
Python environment has a damaged torch package, and an isolated recovery build
failed on incompatible CUDA/CCCL headers before module link. Therefore pixel,
NaN/Inf, and gradient parity are **NOT RUN**. Status: **NOT PASSED**.

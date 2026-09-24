# REFERENCE_SEMANTICS variant

This directory is isolated from the historical model. `topology.py` reproduces
the pinned Graphdeco clone and split child construction, split-parent removal,
and optimizer-state migration while retaining the current gsplat parameter
layout.

Pinned source: `graphdeco-inria/gaussian-splatting` commit
`54c035f7834b564019656c3e3fcc3646292f727d`.

Intentional limits are documented in the C0 report: the paired factorial keeps
the project's masks and schedule fixed to isolate split and Adam effects. It
does not claim parity for the project's gradient statistic, thresholding,
pruning cadence, renderer, initialization, or SH storage.

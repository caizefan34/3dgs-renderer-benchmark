# P14 Prior-Art Boundary (preregistered)

## Explicitly NOT claimed as novel (prior art)

- Gaussian grouping / partitioning G = G_A + G_F (Group Training and
  related work partition Gaussians into groups).
- Freezing + caching a subset of Gaussians to save computation.
  Group Training's cached group is REMOVED from the rendered image.
- Opacity-prioritized sampling of active Gaussians (Speedy-Splat style
  opacity-ordered work reduction is prior art).
- Visible-only parameter updates (visibility-masked update rules are
  standard in official 3DGS code paths).
- Sparse / selective optimizer updates: Sparse Adam, gsplat
  SelectiveAdam (Taming3DGS) are prior art, benchmarked as prior art.
- Asynchronous primitive update of cached components (async caches are
  prior art in rendering systems).

## P14's single tested distinction

Frozen G_F must still FULLY participate in the rendered image
I = R(G_A, G_F) - never R(G_A) - while optimization-induced stationarity
(frozen params + frozen Adam state within a micro-epoch) is exploited to
compress forward, backward, and optimizer work. The novelty status of
this exact compiled-frozen-context mechanism is UNPROVEN by P14; P14
only measures whether the oracle upper bound can pass the frozen gates.

## Status

NOVELTY_STATUS = UNPROVEN. No novelty claim is made by P14 regardless of
outcome. No speedup claim beyond the measured oracle numbers. The
masked-optimizer benchmark (SelectiveAdam / equivalent) is prior art and
is NOT a P14 contribution.

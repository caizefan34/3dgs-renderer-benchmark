# R6-B — rasterizer persistent-gradient prototype

This prototype touches only the unpacked 3DGS rasterizer backward outputs:
`v_means2d`, `v_conics`, `v_colors`, `v_opacities`, and `v_means2d_abs`.
Projection, SH, parameter `.grad`, optimizer, and training math are untouched.

Build an isolated source tree; do not patch the baseline installation:

```bash
python experiments/r6/r6_b/prepare_r6b_source.py \
  --source ~/.local/lib/python3.10/site-packages/gsplat \
  --output /tmp/r6b/gsplat
PYTHONPATH=/tmp/r6b:$PYTHONPATH python experiments/r6/r6_b/run_r6b.py ...
```

`r6b_set_mode(1)` enables B0 (persistent capacity-managed buffers plus a full
active-range clear). `r6b_set_mode(2)` enables B1: the next iteration clears
only the prior iteration's `flatten_ids`, then the current IDs become its next
prior set. `r6b_invalidate()`
must be called after clone, split, or prune; it makes the following backward
perform a full clear.  Packed rendering remains baseline by construction.

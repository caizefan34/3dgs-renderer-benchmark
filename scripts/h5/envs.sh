#!/bin/bash
for E in strong_native anysplat chorus vomp; do
  PY=~/miniforge3/envs/$E/bin/python
  echo "== $E =="
  $PY - <<'PYEOF' 2>&1 | tail -20
import torch, importlib.util, sys
try:
    print("torch", torch.__version__, torch.version.cuda)
    import gsplat
    print("gsplat", getattr(gsplat,"__version__","?"), "expt", end=" ")
    try:
        import gsplat.experimental.render.functional.gaussian_inference as gi
        print("YES", gi._cull_gaussians_batched is not None)
    except Exception as e:
        print("NO", repr(e)[:120])
except Exception as e:
    print("FAIL", repr(e)[:120])
PYEOF
done
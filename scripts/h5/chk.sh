#!/bin/bash
for E in strong_native anysplat nerficg vomp chorus; do
  echo "== $E =="
  ~/miniforge3/envs/$E/bin/python -c 'import gsplat, torch; from gsplat.cuda._wrapper import fully_fused_projection,isect_tiles,isect_offset_encode; print("gsplat",gsplat.__version__ if hasattr(gsplat,"__version__") else "ok","torch",torch.version.cuda)' 2>&1 | tail -1
done
#!/bin/bash
echo "== ENVS =="
ls ~/miniforge3/envs 2>/dev/null
echo "== BASE =="
~/miniforge3/bin/python -c 'import torch;print("torch",torch.__version__,torch.version.cuda,torch.cuda.is_available())' 2>&1 | tail -1
echo "== each env =="
for e in ~/miniforge3/envs/*; do
  echo "== ${e##*/} =="
  "$e/bin/python" -c 'import torch;print("torch",torch.__version__,torch.version.cuda,torch.cuda.is_available())' 2>&1 | tail -1
done
echo "== repo .so =="
find ~/3dgs-renderer-benchmark -name '*.so' 2>/dev/null | grep -i gsplat | head
echo "== prior scalar build =="
ls -la /tmp/higs_scalar_adjoint_build_20260920_r2 2>/dev/null | head
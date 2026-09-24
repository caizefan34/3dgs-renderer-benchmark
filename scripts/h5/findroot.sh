#!/bin/bash
for d in /tmp/higs_h3_fwd_1a_source /tmp/higs_scalar_adjoint_source; do
  echo "== $d =="
  ls -d "$d/gsplat/experimental/render/functional" 2>/dev/null
  find "$d" -name gaussian_inference.py -path '*experimental*' 2>/dev/null | head -3
done
echo "== prior r_memory sys.path =="
grep -nE 'sys.path|importlib' /tmp/h3_fwd_1a_r_memory.py 2>/dev/null | head
echo "== oracle import context =="
grep -rnE 'sys.path|gsplat.experimental' /tmp/higs_h3_fwd_1a_source/gsplat/experimental/render/functional/gaussian_inference.py 2>/dev/null | head
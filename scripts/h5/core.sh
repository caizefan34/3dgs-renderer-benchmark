#!/bin/bash
SRC=/tmp/higs_h3_fwd_1a_source
echo "== gsplat/__init__.py head =="
sed -n '1,40p' $SRC/gsplat/__init__.py 2>/dev/null
echo "== cuda/wrapper imports =="
grep -nE '^import |^from |torch\.ops' $SRC/gsplat/cuda/__init__.py 2>/dev/null | head
echo "== who loads torch.ops.gsplat =="
grep -rnE 'torch\.ops\.load_library|gsplat\.csrc|load .*so' $SRC/gsplat/__init__.py $SRC/gsplat/base.py 2>/dev/null | head
echo "== existing core so =="
ls -la /tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so 2>/dev/null
echo "== experimental gaussian_inference imports =="
grep -nE '^import|^from' $SRC/gsplat/experimental/render/functional/gaussian_inference.py | head -20
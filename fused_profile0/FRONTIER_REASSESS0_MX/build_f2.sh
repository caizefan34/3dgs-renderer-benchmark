#!/bin/bash
# F2 exact/approx SFU variants for blend_backward_cu gaussian.
# Variants: r0 (expf, production), expf_fast (__expf approx intrinsic), exp2f (exp2f(x*M_LOG2E) exact-identity)
cd /mnt/storage_pool/liaoyuanjun/fused_profile0/FRONTIER_REASSESS0_MX
source run_env.sh
OBJ=build/temp.linux-x86_64-cpython-310/mnt_storage_pool_liaoyuanjun_fused_profile0_FRONTIER_REASSESS0_MX/rasterization/src
LIB=build/lib.linux-x86_64-cpython-310/_C_frontier0.cpython-310-x86_64-linux-gnu.so
H=rasterization/include/depreg_probe.h
build_variant() {
  local tag="$1"; local define="$2"
  if [ -n "$define" ]; then printf "#pragma once\n#define %s 1\n" "$define" > $H
  else printf "#pragma once\n" > $H; fi
  rm -f $OBJ/backward.o _C_frontier0.so $LIB
  MAX_JOBS=1 NVCC_EXTRA="-Xptxas -v " $PY setup_depreg.py build_ext --inplace > logs/build_f2_${tag}.log 2>&1
  local ec=$?
  if [ -f "$LIB" ]; then cp -f "$LIB" _C_frontier0_${tag}.so; fi
  echo "TAG $tag exit=$ec so=$([ -f _C_frontier0_${tag}.so ] && echo yes || echo NO)"
}
# rename module to _C_frontier0 for this lane
sed -i "s/_C_depreg0/_C_frontier0/" setup_depreg.py
build_variant r0 ""
build_variant expf_fast "FRONTIER_EXPF_FAST"
build_variant exp2f "FRONTIER_EXPF_EXP2F"
echo "ALL_F2_BUILDS_DONE"

#!/bin/bash
cd /mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_DEPREG0_MX
source run_env.sh
OBJ=build/temp.linux-x86_64-cpython-310/mnt/storage_pool_liaoyuanjun_fused_profile0_BWD_DEPREG0_MX/rasterization/src
LIB=build/lib.linux-x86_64-cpython-310/_C_depreg0.cpython-310-x86_64-linux-gnu.so
H=rasterization/include/depreg_probe.h
build_variant() {
  local tag="$1"; local define="$2"; local maxreg="$3"
  if [ -n "$define" ]; then printf "#pragma once\n#define %s 1\n" "$define" > $H
  else printf "#pragma once\n" > $H; fi
  rm -f $OBJ/backward.o _C_depreg0.so $LIB
  local ne="-Xptxas -v"
  [ -n "$maxreg" ] && ne="-Xptxas -v --maxrregcount=$maxreg"
  MAX_JOBS=1 NVCC_EXTRA="$ne" $PY setup_depreg.py build_ext --inplace > build/logs/build_${tag}.log 2>&1
  local ec=$?
  if [ -f "$LIB" ]; then cp -f "$LIB" _C_depreg0_${tag}.so; fi
  local regs=$(grep "blend_backward_cu" -A2 build/logs/build_${tag}.log | grep -oE "Used [0-9]+ registers" | head -1)
  local spill=$(grep "blend_backward_cu" -A2 build/logs/build_${tag}.log | grep -oE "[0-9]+ bytes spill [a-z]+" | head -2 | tr "\n" " ")
  echo "TAG $tag exit=$ec so=$([ -f _C_depreg0_${tag}.so ] && echo yes || echo NO) regs=[$regs] spill=[$spill]"
}
build_variant r0 "" ""
build_variant dep "BWD_DEPREG_NORECUR" ""
build_variant nocolor "BWD_DEPREG_NOCOLOR" ""
build_variant noopacity "BWD_DEPREG_NOOPACITY" ""
build_variant noconic "BWD_DEPREG_NOCONIC" ""
build_variant occ32 "" "32"
build_variant occ64 "" "64"
echo "ALL_FAST_BUILDS_DONE"
ls -la _C_depreg0*.so

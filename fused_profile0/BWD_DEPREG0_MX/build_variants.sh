cd /mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_DEPREG0_MX
source run_env.sh
OBJ=build/temp.linux-x86_64-cpython-310/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_DEPREG0_MX/rasterization/src
BIND=build/temp.linux-x86_64-cpython-310/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_DEPREG0_MX/torch_bindings
build_one() {
  local tag="$1"; local extra="$2"
  rm -f $OBJ/backward.o $BIND/bindings.o
  rm -f _C_depreg0_${tag}.so _C_depreg0.so
  echo "=== BUILD tag=$tag extra=[$extra] ==="
  DEPREG_TAG="$tag" NVCC_EXTRA="-Xptxas -v $extra" $PY setup_depreg.py build_ext --inplace > build/logs/build_${tag}.log 2>&1
  local ec=$?
  local so=$(ls _C_depreg0_${tag}.so 2>/dev/null)
  local regs=$(grep -E "blend_backward_cu" -A2 build/logs/build_${tag}.log | grep -oE "Used [0-9]+ registers" | head -1)
  local spill=$(grep -E "blend_backward_cu" -A2 build/logs/build_${tag}.log | grep -oE "[0-9]+ bytes spill [a-z]+" | head -2 | tr "\n" " ")
  echo "  exit=$ec so=$so regs=[$regs] spill=[$spill]"
}
build_one 0 ""
build_one dep "-DBWD_DEPREG_NORECUR=1"
build_one nocolor "-DBWD_DEPREG_NOCOLOR=1"
build_one noopacity "-DBWD_DEPREG_NOOPACITY=1"
build_one noconic "-DBWD_DEPREG_NOCONIC=1"
build_one occ32 "--maxrregcount=32"
build_one occ48 "--maxrregcount=48"
build_one occ64 "--maxrregcount=64"
echo "ALL_BUILDS_DONE"
ls -la _C_depreg0*.so

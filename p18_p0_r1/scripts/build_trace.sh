#!/bin/bash
# P18-P0-R1: build the trace-instrumented extension.
#
# ext.cpp / MacroTileRasterize.cu use quoted includes that resolve relative to
# their own directory in the frozen tree. We stage the patched copies in
# directories that reproduce that resolution via symlinks, so ONLY the two
# source paths in build.ninja change (no rule/cflags change -> the other 12
# objects stay cached).
set -euo pipefail

FROZEN=/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/experimental/render/kernels/cuda
SRCDIR=/root/epic05_p18_r1/csrc_trace
EXTROOT=/root/epic05_p18_r1/extroot
RASTROOT=/root/epic05_p18_r1/rastroot
BLD=/root/epic05_p18_r1/build_trace
CACHE=/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda

echo "=== inputs ==="
sha256sum "$SRCDIR/ext.cpp" "$SRCDIR/MacroTileRasterize.cu"

# stage ext.cpp: quoted "csrc/gaussian_inference/*.h" must resolve
# -> EXTROOT/csrc -> frozen kernels/cuda/csrc
mkdir -p "$EXTROOT"
ln -sfn "$FROZEN/csrc" "$EXTROOT/csrc"
cp "$SRCDIR/ext.cpp" "$EXTROOT/ext.cpp"

# stage MacroTileRasterize.cu: quoted same-dir headers must resolve
# -> mirror every sibling of the original .cu (except the .cu itself)
mkdir -p "$RASTROOT"
for f in "$FROZEN"/csrc/gaussian_inference/*; do
    b=$(basename "$f")
    [ "$b" = "MacroTileRasterize.cu" ] && continue
    ln -sfn "$f" "$RASTROOT/$b"
done
cp "$SRCDIR/MacroTileRasterize.cu" "$RASTROOT/MacroTileRasterize.cu"

echo "=== staged layout ==="
ls -la "$EXTROOT"
ls -la "$RASTROOT"
sha256sum "$EXTROOT/ext.cpp" "$RASTROOT/MacroTileRasterize.cu"

rm -rf "$BLD"
cp -a "$CACHE" "$BLD"
cd "$BLD"

sed -i "s|^build ext.o: compile .*\$|build ext.o: compile $EXTROOT/ext.cpp|" build.ninja
sed -i "s|^build MacroTileRasterize.cuda.o: cuda_compile .*\$|build MacroTileRasterize.cuda.o: cuda_compile $RASTROOT/MacroTileRasterize.cu|" build.ninja

echo "=== patched build lines ==="
grep -n '^build ext.o\|^build MacroTileRasterize' build.ninja

NINJA=$(command -v ninja || echo /root/miniforge3/envs/gsplat/bin/ninja)
echo "=== ninja: $NINJA ==="
"$NINJA" -j4

echo "=== outputs ==="
ls -la ext.o MacroTileRasterize.cuda.o experimental_gaussian_render_inference_scene_cuda.so
sha256sum experimental_gaussian_render_inference_scene_cuda.so ext.o MacroTileRasterize.cuda.o
echo "=== exported trace symbol ==="
nm -D --defined-only experimental_gaussian_render_inference_scene_cuda.so | grep -i p18_trace_set || echo "SYMBOL NOT FOUND"
echo "BUILD_TRACE_OK"

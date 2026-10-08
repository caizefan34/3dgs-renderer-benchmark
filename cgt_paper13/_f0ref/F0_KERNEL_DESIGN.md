# CGT-F0 Kernel Design

## Goal
One genuinely fused GPU path for the CGT mixed workload: load compressed CGT
representation -> evaluate student -> rigorous-safe dispatch -> canonical
fallback if needed -> composite -> write output, in the smallest practical
number of kernels, avoiding large HBM intermediates.

## Kernels (f0_kernels.cu, CUDA 11.8, sm_80)
Three templated kernels (float32 fast path, float64 correctness cross-check),
each launched as a single grid over all tiles:

1. \`f0_teacher_kernel\` — canonical path. One block per tile, 256 threads =
   one per 16x16 pixel. Each thread front-to-back alpha-composites the tile's
   M Gaussians -> (E_rgb, T) per pixel.
2. \`f0_student_kernel\` — CGT fast path. One block per tile, 256 threads. Each
   thread evaluates the rank-R (R=4) CGT mixture -> (E_rgb, T) per pixel.
3. \`f0_fused_kernel\` — PRIMARY mixed path. One block per tile, 256 threads.
   Per-BLOCK UNIFORM branch on \`safe_label[t]\`: safe -> student path, else ->
   teacher path. Because the decision is constant across the block, there is NO
   warp divergence — the whole block takes one path.

## Work unit and queue
- Work unit = one gsplat tile (the established CGT evaluation unit).
- The queue is the tile list itself: grid = n_tiles blocks. Large batch (150
  tiles) amortizes launch overhead to a single launch per path.
- Per-tile Gaussian data is flattened into one contiguous array
  (means2d/conics/opac/colors) with \`tile_off\` offsets; per-tile coeffs p[R,10]
  and \`safe_label\` are compact per-tile arrays.

## Memory strategy (no giant HBM record buffer)
- Gaussians are cached in SHARED MEMORY in contiguous depth-ordered chunks
  (CHUNK=512 -> 18 KB) so global HBM traffic is ~M floats/tile, not ~M*256.
  Front-to-back order is preserved across chunks (T carries over), so the
  composite is bit-faithful to the sequential numpy reference.
- Student coeffs (R*10 floats) cached in shared memory.
- Centered pixel coords px[256,2] are IDENTICAL for every tile (every tile is a
  16x16 grid centered on its own mean), so ONE shared px array serves all blocks.
- Only the final (n_tiles,256,4) output is written to HBM. No intermediate
  teacher/student tensor is materialized.

## Numerical fidelity
- Math is bit-faithful to \`tile_teacher\` / \`predict_and_vjp\`.
- Softmax uses max-subtraction (numerically stable; matches scipy softmax),
  preventing fp32 underflow-to-NaN when all logits are very negative.
- Alpha clip [0,0.999] and POWER_CLIP=-4.5 match the frozen teacher exactly.
- Correctness gate: max_abs_RGB <= 2e-5 (preregistered). Measured: f64 ~1e-15,
  f32 ~1.3e-6. PASS.

## Dispatch / certificate semantics
- The rigorous certificate (interval-arithmetic branch-and-bound) is computed
  OFFLINE on CPU during prepare; the kernel receives only the per-tile
  \`safe_label\`. In-kernel dispatch = one int read + a per-block uniform branch.
- Measured dispatch overhead ~0 (the single fused launch is marginally cheaper
  than a two-queue student+teacher split, which saves one launch).

## Compilation
\`nvcc -std=c++17 -O3 -arch=sm_80 -Xcompiler -fPIC -Xptxas -v --shared\`.
Compile-time attributes: 30-40 registers/thread, 0 spill stores/loads. Dynamic
shared memory 18 KB (teacher/fused) or 160 B (student). Theoretical occupancy
~75% (teacher/fused, register-limited) / ~100% (student).

# RASTER-REBASE0 build-compat & harness patches (no algorithm change)

## H (FasterGSFused) setup.py
- c++ standard c++17 -> c++20; added /Zc:preprocessor (CUDA 13.3 CCCL requires standard-conforming preprocessor; torch 2.13 headers require C++20).

## M (diff-gaussian-rasterization_faster) setup.py
- CXX_FLAGS -> ["-O3","-std=c++20","/Zc:preprocessor"]; NVCC_FLAGS added -std=c++20 and -Xcompiler=/Zc:preprocessor.

## M csrc/cuda_rasterizer/render.cu
- CUDA 13.3 CUB removed standalone cub::Max(); added semantics-preserving int max reducer:
  struct RasterRebase0MaxOp { __host__ __device__ int operator()(const int&a,const int&b)const{return a>b?a:b;} };
  BlockReduce(...).Reduce(contrib, cub::Max()) -> RasterRebase0MaxOp().

## common_host/scripts/bench_one.py (harness)
- Direct _C calls for both backends (bypasses Framework/M-wrapper coupling).
- H: w2c[2]=[0,0,-1,4], log-scale, sigmoid-opacity, focal/center camera; params cloned per iter (H backward mutates via fused Adam).
- M: viewmatrix + proj=proj@viewmatrix, linear scale/opacity, tanfov camera; Buffer.curr_offset.fill_(0) per call.
- CUDA-event timing (fwd/bwd split), median over N iters.

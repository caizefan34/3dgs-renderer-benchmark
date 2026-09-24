# H1 Runtime Provenance / ABI Repair

## Result

`H1_BASELINE_IDENTITY = UNRESOLVED`.

The recoverable original in-tree candidate is `/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/gsplat/csrc.so`, SHA-256 `40a009837e75620454059cfdbb5088cc08423f281e4c734da3053b727d39df50`. It is not source-matched to committed `77ab983`.

The committed Python wrapper declares `fully_fused_projection(means,covars,quats,scales,viewmats,Ks,...,packed,sparse_grad,calc_compensations,camera_model,opacities)`. The recovered binary registered `projection_ewa_3dgs_fused(means,covars,quats,scales,opacities,viewmats,Ks,...,calc_compensations,camera_model)`. Thus position 4 is `viewmats` in Python but `opacities` in C++, with all following arguments shifted. It cannot execute the committed H1 source.

The only Torch cache candidate, `/home/liaoyuanjun/.cache/torch_extensions/py310_cu128/gsplat_cuda/gsplat_cuda.so` (SHA-256 `f6fd94d734759ff03d2bc4140611e4cbb206de0a1a9118fb12a580f76b7f19f3`), has a build.ninja proving all sources came from `/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153`; it was created at 20:40 after H1 and is not the original H1 binary.

No source-matched original binary was recovered. Therefore no A/B/C fixture or relabeling is valid. The 1.2B failure is `ROOT_CAUSE_LIKELY`: manually loading the later historical-tree JIT binary through incompatible source/wrapper APIs.

The required isolated build was attempted with `TORCH_EXTENSIONS_DIR=/tmp/h1_identity_clean_build` and exact commit source in `/tmp/h1_identity_77_source`. It failed before compilation because this H1 environment lacks `ninja`; no existing binary was reused.

#!/bin/bash
echo "== existing experimental .so under /tmp =="
find /tmp -name 'experimental_gaussian_render_inference*.so' 2>/dev/null
echo "== build_params.json of r2 =="
cat /tmp/higs_scalar_adjoint_build_20260920_r2/experimental_gaussian_render_inference_scene_cuda/build_params.json 2>/dev/null | head -40
echo "== ninja in envs =="
for E in anysplat chorus; do printf "%s: " $E; ~/miniforge3/envs/$E/bin/python -c "import ninja;print('ninja',ninja.__version__)" 2>&1 | tail -1; done
echo "== gaussian_inference_ops =="
grep -nE 'load_inline|load\(|name=|sources|extra_cuda|with_cuda|archive_path|build_dir' /tmp/higs_h3_fwd_1a_source/gsplat/experimental/render/kernels/gaussian_inference_ops.py 2>/dev/null | head -30
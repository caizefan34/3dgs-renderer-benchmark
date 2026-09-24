#!/bin/bash
export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:$PATH
/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python - <<'PY'
import sys, importlib.util
import torch  # ensures libc10/torch libs loaded so the .so can dlopen
boot="/tmp/p3h_boot"
if boot not in sys.path: sys.path.insert(0, boot)
EXT_NAME="experimental_gaussian_render_inference_scene_cuda"
so="/mnt/storage_pool/liaoyuanjun/higs_p3h_cache/V0/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
spec=importlib.util.spec_from_file_location(EXT_NAME, so)
ext=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)
sys.modules["gsplat.experimental.render.kernels.csrc"]=ext
print("has higs_rasterize_backward:", hasattr(ext,"higs_rasterize_backward"))
print("importing gaussian_inference ...")
from gsplat.experimental.render.functional import gaussian_inference as gi
print("OK gaussian_inference imported; backend available:", gi._higs_backend_available())
print("attr ops:", [x for x in dir(ext) if 'higs' in x])
PY
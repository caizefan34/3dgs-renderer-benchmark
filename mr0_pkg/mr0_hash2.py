import hashlib
files = {
 "c0_ext": "/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so",
 "core":   "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so",
}
for k,p in files.items():
    h = hashlib.sha256(open(p,"rb").read()).hexdigest()
    import os
    print(k, h, os.path.getsize(p))

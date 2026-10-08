import os, glob, subprocess
print("===== PyInit symbols in prebuilt .so files =====", flush=True)
for so in [
  "/root/epic05_build_cache/b1a_ext/gsplat_cuda_final30k/gsplat_cuda_final30k.so",
  "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so",
  "/root/epic05_build_cache/b1_ext/gsplat_cuda_b1clean/gsplat_cuda_b1clean.so",
  "/root/epic05_build_cache/scene/gsplat_scene_cuda/gsplat_scene_cuda.so",
  "/root/epic05_frozen/gsplat-true-accutile-v153/gsplat/csrc.so",
]:
    if not os.path.exists(so):
        print("MISSING", so, flush=True); continue
    try:
        out = subprocess.check_output(["grep","-a","-o","PyInit_[A-Za-z0-9_]*",so], text=True)
        syms = sorted(set(out.split()))
        print(os.path.basename(so), "->", syms, flush=True)
    except Exception as e:
        print(os.path.basename(so), "ERR", e, flush=True)
print("===== stale csrc in true_accutile tree? =====", flush=True)
for p in glob.glob("/root/epic05_frozen/gsplat-true-accutile-v153/gsplat/csrc*"):
    print("  ", p, os.path.getsize(p), flush=True)
print("===== _backend.py 150-230 =====", flush=True)
print(open("/root/epic05_frozen/gsplat-true-accutile-v153/gsplat/cuda/_backend.py").read().splitlines()[149:230].__str__()[:3500], flush=True)
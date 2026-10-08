import os, sys
os.environ["TORCH_EXTENSIONS_DIR"] = "/root/gsc0_ext"
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
# What does the freshly-built .so link against?
import glob
sos = glob.glob("/root/gsc0_ext/gsplat_cuda/*.so")
print("built sos:", sos, flush=True)
for so in sos:
    import subprocess
    try:
        out = subprocess.check_output(["readelf","-d",so], text=True, stderr=subprocess.STDOUT)
        for ln in out.splitlines():
            if "NEEDED" in ln or "libcudart" in ln:
                print("  ", so.split("/")[-1], ln.strip(), flush=True)
    except Exception as e:
        print("  readelf fail", e, flush=True)
# Where is libcudart?
print("libcudart.so.11.0 present?", os.path.exists("/usr/local/cuda/lib64/libcudart.so.11.0"), flush=True)
print("libcudart.so present?", os.path.exists("/usr/local/cuda/lib64/libcudart.so"), flush=True)
for p in glob.glob("/usr/local/cuda/lib64/libcudart*"):
    print("  cuda lib:", p, flush=True)
for p in glob.glob("/root/miniforge3/envs/gsplat/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart*"):
    print("  pip cuda runtime:", p, flush=True)
# torch's bundled cuda
import torch as T
print("torch lib cuda:", os.path.join(os.path.dirname(T.__file__),"lib"), "libcudart" , glob.glob(os.path.join(os.path.dirname(T.__file__),"lib","libcudart*")), flush=True)
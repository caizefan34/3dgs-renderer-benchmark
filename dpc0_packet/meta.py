import json, hashlib, subprocess, os
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()
GI="/root/epic05_dpc0/src_full/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference"
DPC_SO="/root/.cache/torch_extensions/py312_cu128/dpc0_gaussian_render_inference_scene_cuda/dpc0_gaussian_render_inference_scene_cuda.so"
C0_SO="/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
CORE_SO="/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"
# source tree hash (sorted sha256 of all DPC + relevant C0 sources)
srcs=["dpc_stage1.cu","dpc_stage1.h","HigsNativeBackwardDpc.cu","HigsNativeBackwardDpc.h"]
import glob
hs={}
for s in srcs:
    hs[s]=sha(os.path.join(GI,s))
# also ext_dpc.cpp + build_dpc.py
CUDA="/root/epic05_dpc0/src_full/gsplat/experimental/render/kernels/cuda"
hs["ext_dpc.cpp"]=sha(os.path.join(CUDA,"ext_dpc.cpp"))
hs["build_dpc.py"]=sha(os.path.join(CUDA,"build_dpc.py"))
tree=hashlib.sha256()
for k in sorted(hs): tree.update(k.encode()); tree.update(hs[k].encode())
def sh(*a):
    try: return subprocess.check_output(a,text=True).strip()
    except Exception as e: return "ERR:"+str(e)[:80]
nvcc=sh("/usr/local/cuda-12.8/bin/nvcc","--version")
gcc=sh("gcc","--version")
cub=sh("bash","-c","grep -rh 'CCCL_VERSION\\|CUB_VERSION' /usr/local/cuda/include/cub/version.cuh 2>/dev/null | head -2; cat /usr/local/cuda/include/cub/version.cuh 2>/dev/null | grep -i version | head -3")
out={
 "dpc_so_sha256": sha(DPC_SO),
 "c0_so_sha256": sha(C0_SO),
 "core_so_sha256": sha(CORE_SO),
 "c0_expected_sha256":"d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11",
 "source_tree_sha256": tree.hexdigest(),
 "source_files": hs,
 "nvcc": nvcc.splitlines()[-1] if nvcc.startswith("nvcc") else nvcc,
 "gcc": gcc.splitlines()[0] if gcc else gcc,
 "cub": cub,
 "dpc_so_path": DPC_SO,
}
print(json.dumps(out,indent=2))
# write to artifacts
os.makedirs("/root/epic05_dpc0/artifacts",exist_ok=True)
open("/root/epic05_dpc0/dpc_src/build_meta.json","w").write(json.dumps(out,indent=2))
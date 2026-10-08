import platform, sys, json, hashlib, os, glob
out={}
out["python"]=sys.version.split()[0]
out["platform"]=platform.platform()
try:
    import torch
    out["torch"]=torch.__version__
    out["torch_cuda"]=torch.version.cuda
    out["cuda_available"]=torch.cuda.is_available()
    if torch.cuda.is_available():
        out["gpu"]=torch.cuda.get_device_name(0)
        out["gpu_total_mem_MB"]=round(torch.cuda.get_device_properties(0).total_memory/1e6)
        out["capability"]=".".join(map(str,torch.cuda.get_device_capability(0)))
except Exception as e:
    out["torch_error"]=str(e)
for mod in ("gsplat","diff_gaussian_rasterization","fused_ssim"):
    try:
        m=__import__(mod); out[mod+"_version"]=getattr(m,"__version__","(no __version__)")
        out[mod+"_file"]=getattr(m,"__file__",None)
    except Exception as e:
        out[mod+"_error"]=str(e)
out["exe"]=sys.executable
out["hostname"]=platform.node()
print(json.dumps(out,indent=2))

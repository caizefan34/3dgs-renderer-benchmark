import json, os, subprocess, hashlib, socket
env={}
env["host"]="EPIC-05"; env["host_ip"]="8.130.30.251"
try: env["hostname"]=socket.gethostname()
except: pass
def sh(c):
    try: return subprocess.check_output(c,shell=True,stderr=subprocess.DEVNULL,timeout=20).decode().strip()
    except Exception as e: return f"err:{e}"
env["gpu"]=sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1")
env["gpu_count"]=sh("nvidia-smi --query-gpu=index --format=csv,noheader | wc -l")
env["python"]=sh("/root/miniforge3/envs/gsplat/bin/python --version")
env["torch"]=sh("/root/miniforge3/envs/gsplat/bin/python -c \"import torch;print(torch.__version__, torch.version.cuda)\"")
env["gsplat"]=sh("/root/miniforge3/envs/gsplat/bin/python -c \"import gsplat;print(gsplat.__version__)\"")
env["analysis_python"]="/usr/local/bin/python3 (pandas/pyarrow/sklearn, system)"
# frozen trainer + source identity (from D0)
env["frozen_trainer"]="/root/epic05_p13x/publication_trainer_p13x.py"
env["frozen_trainer_p12"]="/root/epic05_p12/publication_trainer_p12.py"
try:
    si=json.load(open("/root/epic05_d0/artifacts/D0_SOURCE_IDENTITY.json"))
    env["source_identity"]=si
except Exception as e: env["source_identity_err"]=str(e)
# D0 provenance
try:
    man=json.load(open("/root/epic05_d0/artifacts/D0_FINAL_MANIFEST.json"))
    env["d0_provenance"]={"keys":list(man.keys())[:20]}
except Exception as e: env["d0_manifest_err"]=str(e)
# scene quality (garden eval from log)
env["garden_final_eval"]="PSNR=29.01 SSIM=0.8814 LPIPS=0.1686 GS=2,568,130 (iter 30000)"
env["protocol"]={"seed":42,"iterations":30000,"densify":"500-15000 @100","opacity_reset":"every 3000 (<15000)",
  "loss":"0.8*L1+0.2*(1-SSIM) fused","eps2d":0.3,"tile":16,"sh_degree":3,
  "scenes":["room","garden","bicycle"],"trainer":"frozen C0 (F9+SCALAR_ADJOINT+H8_MR, PX=2)"}
json.dump(env,open("/root/glm0/artifacts/GLM0_ENVIRONMENT.json","w"),indent=2)
print(json.dumps({k:v for k,v in env.items() if k not in ("source_identity",)},indent=2,default=str)[:1500])

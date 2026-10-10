import os, subprocess, sys
env=dict(os.environ)
env.update({
 "PYTHONDONTWRITEBYTECODE":"1","TMPDIR":"/dev/shm",
 "TORCH_EXTENSIONS_DIR":"/dev/shm/txext","CUDA_HOME":"/usr/local/cuda",
 "PYTHONPATH":"/dev/shm/_ninjapkg:/dev/shm/_richpkg",
 "MAX_JOBS":"8","PATH":"/usr/bin:/bin:/usr/local/cuda/bin:"+env.get("PATH",""),
})
r=subprocess.run(["/home/liaoyuanjun/miniforge3/envs/anysplat/bin/python","-c",
  "import torch,gsplat; from gsplat import rasterization; print('gsplat',gsplat.__version__); print('OK import')"],
  capture_output=True,text=True,env=env)
print("STDOUT",r.stdout[-500:])
print("STDERR",r.stderr[-2000:])
sys.exit(0 if r.returncode==0 else 3)

import subprocess, os
env=dict(os.environ)
env.update({
 "PYTHONDONTWRITEBYTECODE":"1","TMPDIR":"/dev/shm",
 "TORCH_EXTENSIONS_DIR":"/dev/shm/txext","CUDA_HOME":"/usr/local/cuda",
 "MAX_JOBS":"8",
})
# Force rebuild: the csrc.so is stale. gsplat 1.5.3 uses cpp_extension load at import.
# Remove usersite gsplat build cache hint by pointing build to shm.
r=subprocess.run(["/home/liaoyuanjun/miniforge3/envs/anysplat/bin/python","-c",
  "import gsplat; print('imported', gsplat.__version__); "
  "import torch; from gsplat import rasterization; "
  "print('has rasterization')"],
  capture_output=True,text=True,env=env)
print("STDOUT",r.stdout[-800:])
print("STDERR",r.stderr[-1500:])

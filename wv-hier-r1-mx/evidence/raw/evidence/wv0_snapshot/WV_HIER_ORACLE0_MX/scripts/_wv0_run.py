#!/usr/bin/env python3
"""WV0 MX launcher: run a payload with the proven gsplat substrate env.
Usage: wv0_run.py <payload.py> <gpu> <out.json> [wv0_out_dir]
Sets: TMPDIR=/dev/shm, TORCH_EXTENSIONS_DIR=/dev/shm/txext, CUDA_HOME,
PYTHONPATH (ninja+rich modules), PATH incl ~/.local/bin (ninja binary),
PYTHONDONTWRITEBYTECODE=1. Captures stdout/stderr to <out>.log.
"""
import os, subprocess, sys, json, time

NTHREADS = os.environ.get("WV0_THREADS", "8")
ENV_BASE = dict(os.environ)
ENV_BASE.update({
 "PYTHONDONTWRITEBYTECODE":"1","TMPDIR":"/dev/shm",
 "TORCH_EXTENSIONS_DIR":"/dev/shm/txext","CUDA_HOME":"/home/liaoyuanjun/miniforge3/envs/anysplat",
 "PYTHONPATH":"/dev/shm/_ninjapkg:/dev/shm/_richpkg",
 "MAX_JOBS":"8",
 "OMP_NUM_THREADS":NTHREADS,"MKL_NUM_THREADS":NTHREADS,"OPENBLAS_NUM_THREADS":NTHREADS,
 "NUMEXPR_NUM_THREADS":NTHREADS,
 "LD_LIBRARY_PATH":"/home/liaoyuanjun/miniforge3/envs/anysplat/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:"+os.environ.get("LD_LIBRARY_PATH",""),
 "PATH":"/usr/bin:/bin:/home/liaoyuanjun/.local/bin:/home/liaoyuanjun/miniforge3/envs/anysplat/bin:"+os.environ.get("PATH",""),
})
PY="/home/liaoyuanjun/miniforge3/envs/anysplat/bin/python"

def main():
    payload=sys.argv[1]; gpu=sys.argv[2] if len(sys.argv)>2 else "4"
    outp=sys.argv[3] if len(sys.argv)>3 else "/dev/shm/wv0_out/RESULT.json"
    extra=sys.argv[4:]  # forwarded to payload
    env=dict(ENV_BASE); env["CUDA_VISIBLE_DEVICES"]=str(gpu)
    os.makedirs(os.path.dirname(outp),exist_ok=True)
    log=outp+".log"
    t0=time.time()
    with open(log,"w") as lg:
        r=subprocess.run([PY,payload,*extra],env=env,stdout=lg,stderr=subprocess.STDOUT)
    print("WV0_RUN rc",r.returncode,"secs",round(time.time()-t0,1),"log",log)
    sys.exit(r.returncode)

if __name__=="__main__":
    main()




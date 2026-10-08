import subprocess, time, os, sys
os.chdir("/root/A0_P10/A0_P10_PREFLIGHT_KIT/a0preflight")
PY = "/root/miniforge3/envs/gsplat/bin/python"
RUN = "/root/A0_P10/out/run.log"
READY = "/root/A0_P10/out/READY"
def free_mem():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return [int(x) for x in r.stdout.strip().splitlines()]
print("waiting for GPU 0 free, then launching...", flush=True)
t0 = time.time()
while time.time() - t0 < 240:
    m = free_mem()
    if m[0] <= 500:
        print(f"GPU0 free at {m[0]} MiB; launching", flush=True)
        with open(RUN, "w") as f:
            f.write("LAUNCHED via launcher\n")
        os.execv(PY, [PY, "a0p10_run.py", "--out", "/root/A0_P10/out"])
    time.sleep(2)
print("TIMEOUT: GPU0 never free within 240s", flush=True)
with open(READY, "w") as f:
    f.write("TIMEOUT")
sys.exit(2)

import subprocess, glob, os
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout
print("=== kill a0p10 ===")
subprocess.run("pkill -f a0p10_run.py", shell=True)
import time; time.sleep(2)
print("a0p10 pids now:", subprocess.run(["pgrep","-f","a0p10_run.py"],capture_output=True,text=True).stdout or "(none)")
print("=== nvidia compute apps ===")
print(sh("nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader") or "(none)")
print("=== nvidia-smi full (mem) ===")
print(sh("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"))
print("=== per-gpu process listing via nvidia-smi pmon (1s) ===")
print(sh("timeout 3 nvidia-smi pmon -c 1 2>/dev/null") or "(pmon unavailable)")
print("=== large python procs with gpu (ps + /proc/*/environ CUDA_VISIBLE_DEVICES) ===")
for line in sh("ps -eo pid,args | grep -iE 'python' | grep -vE 'grep|jupyter|paimcp|dsw|nohang|supervisor|a0p10'").splitlines():
    parts = line.split(None, 1)
    if not parts: continue
    pid = parts[0]
    try:
        env = open(f"/proc/{pid}/environ","rb").read().decode(errors="ignore")
    except Exception:
        continue
    cvd = [x for x in env.split("\0") if x.startswith("CUDA_VISIBLE_DEVICES=")]
    cmd = parts[1] if len(parts)>1 else ""
    if "python" in cmd:
        print(f"pid={pid} cvd={cvd} cmd={cmd[:80]}")

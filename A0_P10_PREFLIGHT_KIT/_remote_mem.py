import subprocess
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout
print("=== free -h ==="); print(sh("free -h"))
print("=== cpu count ==="); print(sh("nproc"))
print("=== nvidia-smi compute apps (empty if none) ==="); print(sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader") or "(none)")
print("=== any large python procs ==="); print(sh("ps -eo pid,etime,rss,args | grep -iE 'python' | grep -vE 'grep|jupyter|paimcp|dsw|nohang|supervisor' | sort -k3 -rn | head -8"))

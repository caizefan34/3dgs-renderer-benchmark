import subprocess, time
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout
subprocess.run("pkill -f a0p10_run.py", shell=True); time.sleep(2)
print("a0p10:", subprocess.run(["pgrep","-f","a0p10_run.py"],capture_output=True,text=True).stdout or "(none)")
print("=== all gpu mem+util ===")
print(sh("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"))
print("=== compute apps ===")
print(sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader") or "(none)")
print("=== nvidia-smi processes ===")
print(sh("nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader") or "(none)")

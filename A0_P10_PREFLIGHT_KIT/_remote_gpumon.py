import subprocess, time
def smi(*args):
    return subprocess.run(["nvidia-smi"] + list(args), capture_output=True, text=True).stdout
for i in range(3):
    print(f"--- sample {i} ---")
    print(smi("--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"))
    time.sleep(6)
print("=== compute apps ===")
print(smi("--query-compute-apps=pid,used_memory,gpu_uuid", "--format=csv,noheader"))
print("=== a0p10 proc ===")
r = subprocess.run(["pgrep", "-af", "a0p10_run"], capture_output=True, text=True).stdout
print(r.strip() or "none")

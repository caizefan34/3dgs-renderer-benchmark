import subprocess, os
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout
print("=== /root/A0_P10/out/ contents ===")
print(sh("ls -la /root/A0_P10/out/"))
print("=== A0_VERDICT.json ===")
print(sh("cat /root/A0_P10/out/A0_VERDICT.json")[:3000])
print("=== A0_SOURCE_IDENTITY.json (summary) ===")
print(sh("head -c 1500 /root/A0_P10/out/A0_SOURCE_IDENTITY.json"))
print("\n=== tail of run.log (last OOM attempt region) ===")
print(sh("tail -6 /root/A0_P10/out/run.log"))
print("=== remaining a0p10 procs (should be none) ===")
print(sh("pgrep -af a0p10_run | grep -v 'bash -c'") or "(none)")
print("=== gpu state ===")
print(sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader"))

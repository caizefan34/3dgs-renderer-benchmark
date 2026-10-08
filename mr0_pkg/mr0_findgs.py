import hashlib, glob, os, subprocess
# find gsplat package dirs
out = subprocess.run(["bash","-lc","find /root -maxdepth 6 -type d -name gsplat 2>/dev/null; echo '---ENV---'; find /root/miniforge3/envs/gsplat -maxdepth 5 -type d -name gsplat 2>/dev/null"], capture_output=True, text=True)
print(out.stdout)
print("ERR", out.stderr[:500])

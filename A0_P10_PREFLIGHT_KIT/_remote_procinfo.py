import subprocess
def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
print("=== nvidia compute apps (pid mem uuid) ===")
print(sh("nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader"))
print("=== gpu uuid -> index ===")
print(sh("nvidia-smi --query-gpu=index,uuid --format=csv,noheader"))
print("=== top gpu memory processes (ps) ===")
print(sh("ps -eo pid,user,etime,rss,args --sort=-rss | grep -iE 'python|torch|train|a0|n1|gsplat' | grep -v grep | head -15"))
print("=== all 8 gpu mem + util ===")
print(sh("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"))

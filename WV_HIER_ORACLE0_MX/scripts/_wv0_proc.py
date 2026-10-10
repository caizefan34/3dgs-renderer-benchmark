import subprocess
r=subprocess.run(["bash","-lc","ps aux | grep -E '_wv0_campaign_state|_wv0_run' | grep -v grep | awk '{print $2, $12, $13, $14, $15, $16, $17}' | head"],capture_output=True,text=True)
print("PROCS:\n", r.stdout)
r2=subprocess.run(["bash","-lc","nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed -n '5,8p'"],capture_output=True,text=True)
print("GPU 4-7:\n", r2.stdout)
r3=subprocess.run(["bash","-lc","ls -la /dev/shm/wv0_out/logs/ 2>&1; echo '---'; wc -l /dev/shm/wv0_out/logs/*.log 2>&1"],capture_output=True,text=True)
print(r3.stdout)

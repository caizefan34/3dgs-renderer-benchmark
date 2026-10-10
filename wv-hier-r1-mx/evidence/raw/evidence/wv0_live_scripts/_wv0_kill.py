import subprocess
r=subprocess.run(["bash","-lc","ps aux | grep _wv0_campaign_state | grep -v grep | awk '{print $14, $15, $16, $17}'"],capture_output=True,text=True)
print(r.stdout)

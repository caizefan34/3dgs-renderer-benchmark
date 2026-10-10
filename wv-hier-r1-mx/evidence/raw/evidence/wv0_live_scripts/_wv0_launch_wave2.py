import subprocess
PY="/home/liaoyuanjun/miniforge3/envs/anysplat/bin/python"
RUN="/dev/shm/_wv0_run.py"
CAMPAIGN="/dev/shm/_wv0_campaign_state.py"
OUTDIR="/dev/shm/wv0_out/states"
LOGDIR="/dev/shm/wv0_out/logs"
jobs=[("garden__iter15000",4),("garden__iter30000",5),("bicycle__iter5000",6)]
for sid,gpu in jobs:
    out=f"{OUTDIR}/{sid}.pt"; log=f"{LOGDIR}/{sid}.log"
    cmd=f"cd /dev/shm && nohup {PY} {RUN} {CAMPAIGN} {gpu} {OUTDIR}/{sid}.json {sid} 540p 8 {out} 0 > {log} 2>&1 & echo STARTED {sid} gpu{gpu} $!"
    r=subprocess.run(["bash","-lc",cmd],capture_output=True,text=True)
    print(r.stdout.strip())

import subprocess
r=subprocess.run(["/home/liaoyuanjun/miniforge3/envs/anysplat/bin/python","/dev/shm/_wv0_inspect_summary.py"],
   capture_output=True,text=True)
print(r.stdout[:400])

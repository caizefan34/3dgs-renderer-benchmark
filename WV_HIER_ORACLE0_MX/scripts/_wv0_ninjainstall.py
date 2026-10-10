import subprocess
r=subprocess.run(["/home/liaoyuanjun/miniforge3/envs/anysplat/bin/python","-m","pip","install",
   "--target=/dev/shm/_ninjapkg","ninja"],capture_output=True,text=True)
print(r.stdout[-400:]); print("ERR",r.stderr[-300:])
r2=subprocess.run(["/home/liaoyuanjun/miniforge3/envs/anysplat/bin/python","-c",
   "import ninja; print('ninjamod', ninja.BIN_DIR)"],
   capture_output=True,text=True,env={"PYTHONPATH":"/dev/shm/_ninjapkg","PATH":"/usr/bin:/bin"})
print(r2.stdout[-300:]); print("E",r2.stderr[-200:])

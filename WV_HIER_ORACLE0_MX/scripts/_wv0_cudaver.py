import subprocess, glob, os
print("=== nvcc versions ===")
for p in ["/usr/bin/nvcc","/usr/local/cuda*/bin/nvcc"]:
    for f in glob.glob(p):
        r=subprocess.run([f,"--version"],capture_output=True,text=True)
        print(f, r.stdout.strip().split("\n")[-1] if r.stdout else r.stderr[:80])
print("=== cuda dirs ===")
for d in sorted(glob.glob("/usr/local/cuda*")):
    print(d, os.path.exists(d+"/bin/nvcc"))
print("=== conda cuda packages ===")
r=subprocess.run(["bash","-lc","ls /home/liaoyuanjun/miniforge3/envs/anysplat/lib/python3.10/site-packages/ 2>/dev/null | grep -i cuda | head"],capture_output=True,text=True)
print(r.stdout)
# nvidia driver
r=subprocess.run(["bash","-lc","nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"],capture_output=True,text=True)
print("driver", r.stdout.strip())

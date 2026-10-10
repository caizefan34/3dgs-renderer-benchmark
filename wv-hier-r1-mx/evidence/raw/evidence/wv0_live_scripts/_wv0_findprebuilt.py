import subprocess, os, glob
# Look for prebuilt gsplat cuda .so built for torch 2.4 (cu124)
print("=== prebuilt gsplat .so candidates ===")
for pat in ["/dev/shm/txext/**/*.so","/home/liaoyuanjun/higs-13scene/**/*.so",
            "/home/liaoyuanjun/**/*.so"]:
    for p in glob.glob(pat, recursive=True):
        if "gsplat" in p.lower() and ("cuda" in p.lower() or "csrc" in p.lower()):
            print(p, os.path.getsize(p))
# Check gcc versions available
print("=== gcc versions ===")
for g in ["gcc-9","gcc-10","gcc-11","gcc","g++-9","g++-10"]:
    r=subprocess.run(["bash","-lc",f"command -v {g} && {g} --version 2>&1 | head -1"],capture_output=True,text=True)
    print(g, r.stdout.strip().replace("\n"," ") or "MISSING")

import torch, glob, subprocess, os
print("torch:", torch.__version__)
# find libc10_cuda
import torch.cuda
c10 = None
for p in glob.glob("/usr/local/lib/python3.12/site-packages/torch/**/*.so*", recursive=True):
    if "libc10_cuda" in p:
        c10 = p; break
print("libc10_cuda:", c10)
if c10:
    out = subprocess.check_output(["nm","-D",c10],text=True,errors="replace")
    hits=[l for l in out.splitlines() if "c10_cuda_check_implementation" in l]
    print("symbol hits in libc10_cuda:", hits[:4] if hits else "NONE")
# what torch versions are available via pip
try:
    out=subprocess.check_output(["pip","index","versions","torch"],text=True,stderr=subprocess.STDOUT)
    print("pip torch versions (head):", out.splitlines()[:3])
except Exception as e:
    print("pip index err:", str(e)[:120])
# check conda envs
try:
    out=subprocess.check_output(["conda","env","list"],text=True)
    print("conda envs:\n", out)
except Exception as e:
    print("no conda:", str(e)[:80])
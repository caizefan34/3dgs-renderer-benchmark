import os, glob
RUNS="/root/epic05_d0/runs"
print("=== run dirs ===")
for d in sorted(os.listdir(RUNS)):
    p=os.path.join(RUNS,d)
    if os.path.isdir(p):
        npzs=[f for f in os.listdir(p) if f.endswith(".npz")]
        print(d, "->", npzs[:8])
print("=== garden TRACE output ===")
g="/root/glm0/runs/TRACE_garden"
if os.path.isdir(g):
    for f in sorted(os.listdir(g)):
        sz=os.path.getsize(os.path.join(g,f))
        print(f, sz)
else:
    print("not yet created")
print("=== garden log progress ===")
import subprocess

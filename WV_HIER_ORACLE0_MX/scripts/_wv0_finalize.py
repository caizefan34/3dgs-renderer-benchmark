#!/usr/bin/env python3
"""WV-HIER-ORACLE0-MX finalizer: run aggregation + gates + verdict + report + figures,
then copy all deliverables to a local mirror dir. Run on MX after campaign completes.
"""
import sys, os, json, glob, shutil, subprocess, hashlib
OUT = sys.argv[1] if len(sys.argv)>1 else "/dev/shm/wv0_out"
PY = "/home/liaoyuanjun/miniforge3/envs/anysplat/bin/python"

def run(script, *args):
    r=subprocess.run([PY, os.path.join("/dev/shm",script), *args] , capture_output=True, text=True)
    print(f"--- {script} rc={r.returncode} ---")
    print(r.stdout[-1500:])
    if r.returncode!=0: print("ERR", r.stderr[-1500:])
    return r.returncode

# 1. aggregate
run("_wv0_aggregate.py", OUT)
# 2. gates
run("_wv0_gate.py", OUT)
# 3. verdict (writes _verdict.json via the verdict script)
# the verdict script prints; capture to file
r=subprocess.run([PY,"/dev/shm/_wv0_verdict.py",OUT],capture_output=True,text=True)
print("--- verdict ---"); print(r.stdout[-2000:])
if r.returncode!=0: print("VERDICT ERR", r.stderr[-1500:])
# 4. figures
run("_wv0_figures.py", OUT)
# 5. report
run("_wv0_report.py", OUT)
print("FINALIZE_DONE")

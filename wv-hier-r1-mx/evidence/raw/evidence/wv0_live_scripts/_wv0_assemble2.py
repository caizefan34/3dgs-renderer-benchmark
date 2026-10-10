#!/usr/bin/env python3
"""Complete assembly: figures, scripts, logs, RUN_STATUS, report, MANIFEST."""
import sys, os, json, glob, hashlib, shutil, subprocess, time
OUT = sys.argv[1] if len(sys.argv)>1 else "/dev/shm/wv0_out"
DEST = os.path.join(OUT, "WV_HIER_ORACLE0_MX")
PY = "/home/liaoyuanjun/miniforge3/envs/anysplat/bin/python"

# fix protocol/mask filenames
for wrong,right in [("WV0_protocol.json","WV0_PROTOCOL.json"),("WV0_mask_seeds.json","WV0_MASK_SEEDS.json")]:
    wp=os.path.join(DEST,wrong); rp=os.path.join(DEST,right)
    if os.path.exists(wp) and not os.path.exists(rp):
        os.rename(wp,rp); print("renamed",wrong,"->",right)

# copy figures
figs=glob.glob(os.path.join(OUT,"figures","*.png"))
for f in figs: shutil.copy(f, os.path.join(DEST,"figures",os.path.basename(f)))
print("figures:", len(figs))

# copy scripts (all _wv0_*.py in /dev/shm)
scripts=glob.glob("/dev/shm/_wv0_*.py")
for f in scripts:
    try: shutil.copy(f, os.path.join(DEST,"scripts",os.path.basename(f)))
    except Exception as e: print("skip",f,e)
print("scripts:", len(glob.glob(os.path.join(DEST,"scripts","*.py"))))

# copy logs
logs=glob.glob(os.path.join(OUT,"logs","*.log"))
for f in logs:
    try: shutil.copy(f, os.path.join(DEST,"logs",os.path.basename(f)))
    except: pass
print("logs:", len(glob.glob(os.path.join(DEST,"logs","*"))))

# copy driver + run scripts into commands
for f in ["/dev/shm/_wv0_run.py","/dev/shm/_wv0_campaign_driver.sh"]:
    if os.path.exists(f): shutil.copy(f, os.path.join(DEST,"commands",os.path.basename(f)))
# write a commands.sh documenting the run
cmds = """#!/bin/bash
# WV-HIER-ORACLE0-MX command log
# Campaign driver (9 states across GPUs 4-7):
cd /dev/shm && bash /dev/shm/_wv0_campaign_driver.sh
# Per-state invocation:
#   python _wv0_run.py _wv0_campaign_state.py <gpu> <out.json> <state_id> 540p 8 <out.pt> 0
# Analysis:
#   python _wv0_aggregate.py /dev/shm/wv0_out
#   python _wv0_gate.py /dev/shm/wv0_out
#   python _wv0_verdict.py /dev/shm/wv0_out
#   python _wv0_figures.py /dev/shm/wv0_out
#   python _wv0_report.py /dev/shm/wv0_out
"""
open(os.path.join(DEST,"commands","commands.sh"),"w").write(cmds)

# RUN_STATUS.json
verd=json.load(open(os.path.join(OUT,"_verdict.json")))
run_status={
  "campaign":"WV-HIER-ORACLE0-MX",
  "status":"COMPLETE",
  "timestamp":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "backend":"gsplat 1.5.3 (torch 2.4.1+cu124, CUDA 12.4) on MX A100",
  "n_states":9,
  "states":sorted([s for s in os.listdir(os.path.join(OUT,"states")) if s.endswith(".pt")]),
  "granularities":[16,8,4,1],
  "budgets":[0.25,0.40],
  "n_masks":32,
  "cameras_per_state":8,
  "resolution":"540p",
  "verdict":verd["verdict"],
  "interior_pareto":"G4",
  "gate_A_G4_pass":verd["gateA"]["4"]["gateA_pass"],
  "gate_D_pass":verd["gateA"]["4"].get("gateD_pass"),
}
json.dump(run_status, open(os.path.join(DEST,"RUN_STATUS.json"),"w"), indent=2)
print("RUN_STATUS written")

# copy report
if os.path.exists(os.path.join(OUT,"WV0_REPORT.md")):
    shutil.copy(os.path.join(OUT,"WV0_REPORT.md"), os.path.join(DEST,"WV0_REPORT.md"))
    print("report copied")

# MANIFEST.sha256 over all files in DEST (excluding MANIFEST itself)
lines=[]
for root,_,files in os.walk(DEST):
    for fn in sorted(files):
        if fn=="MANIFEST.sha256": continue
        p=os.path.join(root,fn)
        h=hashlib.sha256(open(p,"rb").read()).hexdigest()
        rel=os.path.relpath(p,DEST).replace("\\","/")
        lines.append(f"{h}  {rel}")
lines.sort()
open(os.path.join(DEST,"MANIFEST.sha256"),"w").write("\n".join(lines)+"\n")
print("MANIFEST written:", len(lines), "files")

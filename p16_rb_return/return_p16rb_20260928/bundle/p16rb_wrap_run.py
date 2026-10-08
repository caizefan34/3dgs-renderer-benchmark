#!/usr/bin/env python3
"""Minimal native-run wall-clock wrapper for P16-RB.

This wrapper intentionally does not instrument method internals.
It records command wall time, stdout/stderr, and nvidia-smi state.
"""
import argparse, json, os, shlex, subprocess, time
from pathlib import Path
from datetime import datetime, timezone

def run_text(cmd):
    p=subprocess.run(cmd,shell=True,text=True,capture_output=True)
    return {"returncode":p.returncode,"stdout":p.stdout,"stderr":p.stderr}

ap=argparse.ArgumentParser()
ap.add_argument("--method",required=True)
ap.add_argument("--scene",required=True)
ap.add_argument("--cwd",required=True)
ap.add_argument("--command",required=True)
ap.add_argument("--out",required=True)
ap.add_argument("--log",required=True)
args=ap.parse_args()

before=run_text("nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv,noheader,nounits")
proc_before=run_text("nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits")

start_iso=datetime.now(timezone.utc).isoformat()
t0=time.perf_counter()
p=subprocess.run(args.command,shell=True,cwd=args.cwd,text=True,capture_output=True)
wall=time.perf_counter()-t0
end_iso=datetime.now(timezone.utc).isoformat()

after=run_text("nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv,noheader,nounits")
proc_after=run_text("nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits")

Path(args.log).parent.mkdir(parents=True,exist_ok=True)
Path(args.log).write_text(
    "COMMAND:\n"+args.command+"\n\nSTDOUT:\n"+p.stdout+"\n\nSTDERR:\n"+p.stderr,
    encoding="utf-8"
)
out={
    "method":args.method,"scene":args.scene,
    "cwd":str(Path(args.cwd).resolve()),"command":args.command,
    "start_utc":start_iso,"end_utc":end_iso,
    "command_wall_s":wall,"returncode":p.returncode,
    "gpu_before":before,"gpu_after":after,
    "compute_processes_before":proc_before,"compute_processes_after":proc_after
}
Path(args.out).parent.mkdir(parents=True,exist_ok=True)
Path(args.out).write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps({"method":args.method,"scene":args.scene,"wall_s":wall,"returncode":p.returncode},indent=2))
raise SystemExit(p.returncode)

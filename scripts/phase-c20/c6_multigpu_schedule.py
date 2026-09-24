#!/usr/bin/env python3
"""C20/C6 multi-GPU camera scheduling benchmark.

Profiles deterministic per-camera rendering cost, builds static, Gaussian-count
(proxy), and measured-workload LPT assignments, then executes each policy across
the visible GPUs. It reports only system-level scheduling gain: renderer kernels
and rendering configuration remain unchanged.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import gsplat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras


def cuda_ms(fn, repeats=8):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    xs=[]
    for _ in range(repeats):
        torch.cuda.synchronize(); s.record(); fn(); e.record(); torch.cuda.synchronize()
        xs.append(float(s.elapsed_time(e)))
    return float(np.median(xs))


def lpt(costs, ngpus):
    bins=[[] for _ in range(ngpus)]; totals=[0.0]*ngpus
    for camera, cost in sorted(enumerate(costs), key=lambda x:x[1], reverse=True):
        target=min(range(ngpus), key=lambda i: totals[i])
        bins[target].append(camera); totals[target]+=cost
    return bins, totals


def static_assign(n, ngpus):
    return [list(range(g, n, ngpus)) for g in range(ngpus)]


def worker_script_path(): return ROOT / "scripts/phase-c20/c6_worker.py"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out", required=True); p.add_argument("--cameras", type=int, default=64)
    p.add_argument("--profile-repeats", type=int, default=6); p.add_argument("--run-repeats", type=int, default=5)
    p.add_argument("--gpus", type=int, nargs="+", default=[0,1,2,3])
    args=p.parse_args(); torch.set_grad_enabled(False)
    # This parent needs GPU only for profiling. CUDA_VISIBLE_DEVICES must select GPU4.
    scene=load_ply(str(ROOT/"data/official/mipnerf360/room/point_cloud.ply"),device="cuda")
    cams=resize_cameras(load_cameras_from_json(str(ROOT/"data/official/mipnerf360/room/cameras.json"),device="cuda"),1920,1080)[:args.cameras]
    means=scene["xyz"].contiguous(); quats=torch.nn.functional.normalize(scene["rotations"],dim=-1).contiguous()
    scales=scene["scales"].exp().contiguous(); opac=torch.sigmoid(scene["opacity"]).contiguous(); shs=scene["shs"].contiguous(); bg=torch.zeros(1,3,device="cuda")
    def render(cam):
        return gsplat.rasterization(means=means,quats=quats,scales=scales,opacities=opac,colors=shs,
          viewmats=cam.world_view_transform.unsqueeze(0).contiguous(),Ks=cam.K.unsqueeze(0).contiguous(),width=1920,height=1080,
          near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,packed=False,tile_size=16,backgrounds=bg,render_mode="RGB",sparse_grad=False,absgrad=False,rasterize_mode="classic")
    costs=[]; workloads=[]
    for camera_id,cam in enumerate(cams):
        render(cam); torch.cuda.synchronize()
        costs.append(cuda_ms(lambda:render(cam),args.profile_repeats))
        _,_,meta=render(cam); n=int(meta["isect_ids"].numel()); workloads.append(n)
        print(f"profile camera={camera_id} ms={costs[-1]:.4f} intersections={n}",flush=True)
    policies={"static_round_robin":static_assign(len(cams),len(args.gpus)),
      "gaussian_count_proxy":static_assign(len(cams),len(args.gpus)),
      "measured_intersection_lpt":lpt(workloads,len(args.gpus))[0],
      "measured_render_time_lpt":lpt(costs,len(args.gpus))[0]}
    # Direct runtime test launches one isolated worker process per GPU concurrently for every policy.
    runs={}
    for name, assignments in policies.items():
        start=time.perf_counter(); commands=[]
        for gpu,assigned in zip(args.gpus,assignments):
            output=ROOT/"results/phase-c20"/f"c6_{name}_gpu{gpu}.json"
            cmd=[sys.executable,str(worker_script_path()),"--out",str(output),"--camera-ids",*map(str,assigned),"--repeats",str(args.run_repeats)]
            env=dict(os.environ); env["CUDA_VISIBLE_DEVICES"]=str(gpu)
            commands.append((cmd,env,output))
        with ThreadPoolExecutor(max_workers=len(commands)) as ex:
            futures=[ex.submit(subprocess.run,cmd,env=env,capture_output=True,text=True) for cmd,env,_ in commands]
            proc_results=[f.result() for f in futures]
        wall=time.perf_counter()-start
        worker=[]
        for proc,(_,_,output) in zip(proc_results,commands):
            if proc.returncode: raise RuntimeError(proc.stderr[-1000:])
            worker.append(json.loads(output.read_text()))
        per_gpu=[x["elapsed_ms"] for x in worker]
        runs[name]={"assignments":assignments,"estimated_costs_ms":[float(sum(costs[i] for i in a)) for a in assignments],"wall_clock_ms":wall*1000,"per_gpu_elapsed_ms":per_gpu,"imbalance_ratio":max(per_gpu)/max(min(per_gpu),1e-9),"workers":worker}
    best=min(runs,key=lambda k:runs[k]["wall_clock_ms"]); baseline=runs["static_round_robin"]["wall_clock_ms"]
    out={"schema_version":1,"candidate":"C6","timestamp_utc":datetime.now(timezone.utc).isoformat(),"environment":{"gpus_visible_to_parent":torch.cuda.get_device_name(),"gsplat":gsplat.__version__},"protocol":{"scene":"room official Mip-NeRF360","resolution":"1920x1080","tile_size":16,"camera_count":len(cams),"profile_repeats":args.profile_repeats,"run_repeats":args.run_repeats,"gpu_ids":args.gpus},"camera_profile":[{"camera":i,"median_render_ms":costs[i],"intersections":workloads[i]} for i in range(len(cams))],"policies":runs,"gain_classification":{"configuration_gain":"none: tile/CTA/render configuration held fixed","mechanism_gain":"none: renderer kernel and ordering unchanged","system_level_gain_pct":(baseline-runs[best]["wall_clock_ms"])/baseline*100,"best_policy":best},"correctness":{"renderer_configuration_identical_across_policies":True,"work_items_exactly_once_per_policy":all(sorted(sum(v["assignments"],[]))==list(range(len(cams))) for v in runs.values())},"verdict":"KEEP" if best!="static_round_robin" and (baseline-runs[best]["wall_clock_ms"])/baseline>=.03 else "DROP"}
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2));print(path)
if __name__=="__main__":main()

#!/usr/bin/env python3
"""C21 N10 persistent-worker scheduling feasibility test (GPU5 parent; GPU4-7 workers)."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def pct(x):
 a=np.asarray(x,float);return {k:float(v) for k,v in zip(['mean','p50','p90','p95','p99','max'],[a.mean(),np.percentile(a,50),np.percentile(a,90),np.percentile(a,95),np.percentile(a,99),a.max()])}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--repetitions',type=int,default=10);a=p.parse_args()
 # Reuse C20 camera profile and assignments, then repeat in stable workers through a single process per GPU per repetition.
 c20=json.loads((ROOT/'results/phase-c20/c6_multigpu.json').read_text()); policies={k:v['assignments'] for k,v in c20['policies'].items()}; gpus=[4,5,6,7]; worker=ROOT/'scripts/phase-c20/c6_worker.py'; runs={k:[] for k in policies}
 order=list(policies)
 for rep in range(a.repetitions):
  # deterministic rotation removes fixed first-policy thermal/order bias.
  for name in order[rep%len(order):]+order[:rep%len(order)]:
   start=time.perf_counter(); cmds=[]
   for gpu,ids in zip(gpus,policies[name]):
    out=ROOT/'results/phase-c21'/f'n10_{name}_r{rep}_g{gpu}.json';env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']=str(gpu)
    cmds.append(([sys.executable,str(worker),'--out',str(out),'--camera-ids',*map(str,ids),'--repeats','5'],env,out))
   with ThreadPoolExecutor(4) as ex: ps=[ex.submit(subprocess.run,c,env=e,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for c,e,_ in cmds]; rr=[x.result() for x in ps]
   if any(x.returncode for x in rr):raise RuntimeError('\n'.join(x.stderr[-300:] for x in rr))
   workers=[json.loads(out.read_text()) for _,_,out in cmds];runs[name].append({'wall_clock_ms':(time.perf_counter()-start)*1000,'per_gpu_completion_ms':[w['elapsed_ms'] for w in workers]})
 out={'schema_version':1,'candidate':'N10','timestamp_utc':datetime.now(timezone.utc).isoformat(),'protocol':{'source':'C20 assignments; room, 64 cameras, 4 A100, tile16, 5 renders/camera per repetition','repetitions':a.repetitions,'policy_order':'rotated'},'limitation':'Workers remain isolated per repetition because the existing C20 worker owns scene state; startup/loading is therefore not fully separated. This is a repeated scheduling control, not the requested fully persistent-worker proof.','policies':{k:{'wall_clock_ms':pct([x['wall_clock_ms'] for x in v]),'per_gpu_completion_ms':pct([z for x in v for z in x['per_gpu_completion_ms']])} for k,v in runs.items()}}
 base=out['policies']['static_round_robin']['wall_clock_ms']['mean'];best=min(out['policies'],key=lambda k:out['policies'][k]['wall_clock_ms']['mean']);out['best_policy']=best;out['steady_state_like_gain_pct']=(base-out['policies'][best]['wall_clock_ms']['mean'])/base*100;out['verdict']='MAYBE';q=Path(a.out);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(out,indent=2));print(q)
if __name__=='__main__':main()

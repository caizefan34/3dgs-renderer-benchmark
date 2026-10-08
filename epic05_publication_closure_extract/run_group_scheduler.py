#!/usr/bin/env python3
"""
Grouped EPIC-05 scheduler.
Each comparison group stays on one physical GPU, eliminating cross-device bias
within B1A/C0, A0/A1/A2/C0, eps2d, and multi-seed pairs.
"""
import argparse, json, os, queue, subprocess, threading, time, pathlib, shlex

def gpu_snapshot(gpu, path):
    cmd=["nvidia-smi","-i",str(gpu),
         "--query-gpu=timestamp,index,uuid,name,pstate,memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu",
         "--format=csv,noheader"]
    p=subprocess.run(cmd,capture_output=True,text=True)
    pathlib.Path(path).write_text((p.stdout or "")+(p.stderr or ""),encoding="utf-8")

def run_job(job, gpu, args):
    outdir=pathlib.Path(args.out)/job["run_id"]
    outdir.mkdir(parents=True,exist_ok=True)
    done=outdir/"EPIC05_SUCCESS"
    if done.exists() and not args.rerun:
        print("SKIP", job["run_id"], flush=True); return True

    gpu_snapshot(gpu, outdir/"gpu_before.csv")
    cmd=[args.python,args.trainer,
         "--arm",job["arm"],"--scene",job["scene"],
         "--iterations",str(job.get("iterations",30000)),
         "--outdir",str(outdir),"--gpu",str(gpu),
         "--lpips","--final-eval","all","--checkpoints",
         "--timing-grade","PUBLICATION"]
    if job.get("seed",42)!=42:
        cmd += ["--seed",str(job["seed"])]
    # Always pass eps2d explicitly so provenance is unambiguous.
    if job.get("eps2d") is not None:
        cmd += ["--eps2d",str(job["eps2d"])]

    (outdir/"command.txt").write_text(" ".join(shlex.quote(x) for x in cmd)+"\n",encoding="utf-8")
    env=os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]=str(gpu)
    # Trainer receives --gpu=0 when CUDA_VISIBLE_DEVICES remaps one device.
    # Replace its physical gpu argument with logical 0.
    i=cmd.index("--gpu")+1
    cmd[i]="0"
    t0=time.time()
    with open(outdir/"stdout.log","w",buffering=1) as f:
        p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,text=True)
    gpu_snapshot(gpu, outdir/"gpu_after.csv")
    meta={"run_id":job["run_id"],"group_gpu_physical":gpu,"returncode":p.returncode,
          "wall_launcher_s":time.time()-t0,"job":job}
    (outdir/"epic05_launcher_meta.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    if p.returncode==0 and (outdir/"results.json").exists():
        done.write_text("PASS\n")
        return True
    (outdir/"EPIC05_FAILED").write_text(f"returncode={p.returncode}\n")
    return False

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--trainer",required=True)
    ap.add_argument("--python",default="/root/miniforge3/envs/gsplat/bin/python")
    ap.add_argument("--out",default="/root/3dgs-publication-epic05/full30k")
    ap.add_argument("--gpus",default="0,1,2,3,4,5,6,7")
    ap.add_argument("--rerun",action="store_true")
    args=ap.parse_args()
    m=json.load(open(args.manifest))
    q=queue.Queue()
    for g in m["groups"]: q.put(g)
    failures=[]
    lock=threading.Lock()

    def worker(gpu):
        while True:
            try: group=q.get_nowait()
            except queue.Empty: return
            print(f"[GPU{gpu}] GROUP {group['group_id']} ({len(group['jobs'])} jobs)",flush=True)
            ok=True
            for job in group["jobs"]:
                if not run_job(job,gpu,args):
                    ok=False
                    with lock: failures.append(job["run_id"])
                    # Keep the group on the same GPU, but do not run downstream pair members
                    # after a failure: prevents partial matched comparisons.
                    break
            q.task_done()
            print(f"[GPU{gpu}] GROUP {group['group_id']} {'PASS' if ok else 'FAIL'}",flush=True)

    threads=[threading.Thread(target=worker,args=(int(g),),daemon=False)
             for g in args.gpus.split(",") if g.strip()]
    for t in threads: t.start()
    for t in threads: t.join()
    summary={"failures":failures,"n_failures":len(failures)}
    pathlib.Path(args.out).mkdir(parents=True,exist_ok=True)
    pathlib.Path(args.out,"scheduler_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    raise SystemExit(1 if failures else 0)

if __name__=="__main__": main()

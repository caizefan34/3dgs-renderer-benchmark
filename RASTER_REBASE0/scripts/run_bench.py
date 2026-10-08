import subprocess, json, sys, os, statistics, time

PY=sys.executable
BENCH=r"C:\rb\bench_one.py"
OUT=r"C:\Users\36570\3dgs-renderer-benchmark\RASTER_REBASE0\raw"
os.makedirs(OUT,exist_ok=True)

def run(backend,N,W,H,iters,view):
    t0=time.time()
    p=subprocess.run([PY,BENCH,backend,str(N),str(W),str(H),str(iters),str(view)],
                     capture_output=True,text=True,env={**os.environ,"CUDA_VISIBLE_DEVICES":"0"})
    line=[l for l in p.stdout.splitlines() if "JSON_START" in l]
    if not line:
        print("FAIL",backend,N,view,p.stderr[-400:]); return None
    j=json.loads(line[0].split("JSON_START")[1].split("JSON_END")[0])
    j["wall_s"]=round(time.time()-t0,2)
    return j

def frac(r): return r["bwd_ms"]/(r["fwd_ms"]+r["bwd_ms"])

if __name__=="__main__":
    Ns=[int(x) for x in sys.argv[1].split(",")] if len(sys.argv)>1 else [50000,100000]
    views=[int(x) for x in sys.argv[2].split(",")] if len(sys.argv)>2 else [0,1,2,3]
    iters=int(sys.argv[3]) if len(sys.argv)>3 else 40
    W=H=256
    allrows=[]
    for N in Ns:
        for v in views:
            for b in ["H","M"]:
                r=run(b,N,W,H,iters,v)
                if r is None: continue
                r["frac"]=frac(r); r["N"]=N; r["view"]=v
                allrows.append(r)
                print(f"N={N} view={v} {b}: fwd={r['fwd_ms']:.4f} bwd={r['bwd_ms']:.4f} frac={r['frac']:.3f} inst={r['instances']} buck={r['buckets']} ({r['wall_s']}s)")
    with open(os.path.join(OUT,"bench_grid.json"),"w") as f: json.dump(allrows,f,indent=2)
    # aggregate by backend
    agg={}
    for b in ["H","M"]:
        rows=[r for r in allrows if r["backend"]==b]
        agg[b]=dict(
            n=len(rows),
            fwd_median=statistics.median([r["fwd_ms"] for r in rows]),
            bwd_median=statistics.median([r["bwd_ms"] for r in rows]),
            frac_median=statistics.median([r["frac"] for r in rows]),
            frac_mean=statistics.mean([r["frac"] for r in rows]),
            instances_median=statistics.median([r["instances"] for r in rows]),
        )
    print("\n=== AGGREGATE ===")
    print(json.dumps(agg,indent=2))
    with open(os.path.join(OUT,"bench_agg.json"),"w") as f: json.dump(agg,f,indent=2)

import sys, os, json, time, argparse
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atom0_common as A

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--stage', required=True)
    ap.add_argument('--cam_idx', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--warmup', type=int, default=30)
    ap.add_argument('--samples', type=int, default=20)
    ap.add_argument('--chunk_n_buckets', type=int, default=5000000)
    args = ap.parse_args()

    t0 = time.time()
    trainer, dataset, train_ds, sampler, ad, g = A.build_model(args.config, args.checkpoint, 0)
    n_prim = int(g.means.shape[0])
    view = train_ds[args.cam_idx]
    settings, bg = A.view_settings(trainer, view, g)
    fd = A.call_forward(g, settings, bg)
    print(f"FWD {args.scene}/{args.stage} cam{args.cam_idx}: n_prim={n_prim} "
          f"n_instances={fd['n_instances']} n_buckets={fd['n_buckets']} "
          f"wh={settings.width}x{settings.height}", flush=True)
    if fd['n_instances'] <= 0 or fd['n_buckets'] <= 0:
        raise RuntimeError(f"degenerate forward ninst={fd['n_instances']} nbuckets={fd['n_buckets']}")

    res = A.measure_abc(fd, settings, bg, n_prim, seed=args.seed,
                        chunk_n_buckets=args.chunk_n_buckets,
                        warmup=args.warmup, samples=args.samples,
                        capture_equiv=True, tag=f"{args.scene}/{args.stage}/cam{args.cam_idx}")

    # workload: instances per Gaussian histogram + hot/cold partition
    inst_prim = A.get_instance_prim(fd)  # cuda int32
    cnt = torch.bincount(inst_prim, minlength=n_prim).to(torch.float64)  # per-primitive instance counts
    visible = int((cnt > 0).sum().item())
    ipg = cnt[cnt > 0]
    def q(x, p): return float(torch.quantile(x, p).item()) if x.numel() > 0 else 0.0
    total_inst = float(cnt.sum().item())
    # hot/cold bins by per-primitive instance multiplicity
    bins = [(1,8),(9,32),(33,128),(129,10**9)]
    hc = []
    for lo,hi in bins:
        m = (cnt >= lo) & (cnt <= hi)
        hc.append({"bin": f"{lo}-{hi if hi<10**9 else '+'}",
                   "n_primitives": int(m.sum().item()),
                   "n_instances": float(cnt[m].sum().item()),
                   "instance_fraction": float(cnt[m].sum().item()/total_inst) if total_inst>0 else 0.0})
    # concentration: top 1% / 5% primitives share of instances
    scnt, _ = torch.sort(cnt, descending=True)
    def top_share(k):
        k = min(k, scnt.numel())
        return float(scnt[:k].sum().item()/total_inst) if total_inst>0 else 0.0
    res['workload'] = {
        "n_primitives": n_prim, "n_visible_gaussians": visible,
        "n_instances": int(fd['n_instances']), "n_buckets": int(fd['n_buckets']),
        "instances_per_gaussian_mean": float(ipg.mean().item()) if visible>0 else 0.0,
        "instances_per_gaussian_p50": q(ipg,0.5), "instances_per_gaussian_p90": q(ipg,0.9),
        "instances_per_gaussian_p99": q(ipg,0.99), "instances_per_gaussian_max": float(ipg.max().item()) if visible>0 else 0.0,
        "amplification_factor": round(fd['n_instances']/max(1,visible),3),
        "top1pct_primitive_instance_share": top_share(max(1,int(0.01*visible))),
        "top5pct_primitive_instance_share": top_share(max(1,int(0.05*visible))),
        "hot_cold_bins": hc,
    }
    res['meta'] = {"scene": args.scene, "stage": args.stage, "cam_idx": args.cam_idx,
                   "n_primitives": n_prim, "width": settings.width, "height": settings.height,
                   "warmup": args.warmup, "samples": args.samples,
                   "seed": args.seed, "chunk_n_buckets": args.chunk_n_buckets,
                   "wall_sec": round(time.time()-t0,1)}
    json.dump(res, open(args.out,'w'), indent=2)
    print("WROTE", args.out, "T_A_full=%.4f T_A=%.4f T_B=%.4f T_C=%.4f ms" % (
        res['T_A_full_ms'], res['T_A_ms'], res['T_B_ms'], res['T_C_ms']), flush=True)

if __name__ == '__main__':
    main()

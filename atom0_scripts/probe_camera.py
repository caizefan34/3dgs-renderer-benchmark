import sys, os, json, time, argparse, traceback
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atom0_common as A

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--stage', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=40)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    trainer, dataset, train_ds, sampler, ad, g = A.build_model(args.config, args.checkpoint, 0)
    n_prim = int(g.means.shape[0])
    n = len(train_ds)
    print(f"PROBE {args.scene}/{args.stage}: n_prim={n_prim} n_cameras={n}", flush=True)
    rows = []
    end = min(args.end, n)
    for i in range(args.start, end):
        try:
            view = train_ds[i]
            settings, bg = A.view_settings(trainer, view, g)
            t0=time.time()
            fd = A.call_forward(g, settings, bg)
            dt=time.time()-t0
            rows.append({"cam": i, "ok": True, "n_instances": fd['n_instances'],
                         "n_buckets": fd['n_buckets'], "wh": [settings.width, settings.height],
                         "fwd_sec": round(dt,2)})
            print(f"  cam{i}: OK ninst={fd['n_instances']} nbuckets={fd['n_buckets']} wh={settings.width}x{settings.height} fwd={dt:.1f}s", flush=True)
        except Exception as e:
            rows.append({"cam": i, "ok": False, "err": f"{type(e).__name__}: {str(e)[:120]}"})
            print(f"  cam{i}: FAIL {type(e).__name__}: {str(e)[:120]}", flush=True)
    json.dump({"scene":args.scene,"stage":args.stage,"n_prim":n_prim,"n_cameras":n,"rows":rows},
              open(args.out,'w'), indent=2)
    print("WROTE", args.out, flush=True)

if __name__ == '__main__':
    main()

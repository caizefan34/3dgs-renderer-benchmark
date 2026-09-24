import json
import os

F30K = "/mnt/storage_pool/liaoyuanjun/final30k_runs"
PUB = "/mnt/storage_pool/liaoyuanjun/pub_runs"

for root in (F30K, PUB):
    print(f"== {root}")
    for d in sorted(os.listdir(root)):
        if d.startswith(("a0_", "a1_", "a2_")):
            rj = os.path.join(root, d, "results.json")
            if os.path.exists(rj):
                r = json.load(open(rj))
                t = r["timing"]
                print(f"  {d:16s} wall={t.get('total_wall_s')} iters={t.get('n_iters')} "
                      f"psnr={r['final_eval']['psnr']:.3f} "
                      f"bin={r.get('binary_identity', {}).get('so_sha256', '')[:12]} "
                      f"grade={r.get('timing_grade')} arm={r.get('arm')}")
            else:
                print(f"  {d:16s} (no results.json)")

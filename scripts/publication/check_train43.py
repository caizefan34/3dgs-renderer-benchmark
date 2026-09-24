import json
m = json.load(open('/mnt/storage_pool/liaoyuanjun/pubphase/aggregates/runs_master.json'))
hits = [r for r in m["runs"] if r["scene"] == "train" and r["seed"] == 43]
for r in hits:
    print(r["run_id"], "variant=", r["variant"], "eps2d=", r["eps2d"],
          "contaminated=", r.get("contaminated"), "timing_grade=", r.get("timing_grade"),
          "wall=", r.get("wall_s"))
print("total runs in master:", len(m["runs"]))

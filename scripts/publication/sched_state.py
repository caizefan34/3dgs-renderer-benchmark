import json, glob, os, time

st = json.load(open("/mnt/storage_pool/liaoyuanjun/pub_runs/pub_scheduler_state.json"))
print("STATUS:", st.get("status"))
print("\n== DONE ==", len(st["done"]))
for d in st["done"]:
    print(f"  {d['rid']:22s} gpu={d['gpu']} rc={d['rc']} wall={d.get('wall_s','-')}s "
          f"contam={d.get('contaminated')} note={d.get('note','')}")
print("\n== FAILED ==", len(st["failed"]))
for d in st["failed"]:
    print(f"  {d['rid']:22s} gpu={d['gpu']} rc={d['rc']} wall={d.get('wall_s','-')}s note={d.get('note','')}")
print("\n== ACTIVE ==", len(st["active"]))
for g, a in st["active"].items():
    age_min = (time.time() - a["started"]) / 60
    print(f"  gpu={g} {a['rid']:22s} pid={a['pid']} age={age_min:.1f}min")
print("\n== CONTAMINATED ==")
print(json.dumps(st.get("contaminated", {})))

print("\n== FAILED RUN LOG TAILS ==")
for d in st["failed"]:
    p = f"/mnt/storage_pool/liaoyuanjun/pub_runs/log_{d['rid']}.txt"
    if os.path.exists(p):
        lines = open(p, errors="replace").read().strip().splitlines()
        print(f"---- {d['rid']} (last 6 of {len(lines)}) ----")
        for ln in lines[-6:]:
            print("   ", ln[:160])

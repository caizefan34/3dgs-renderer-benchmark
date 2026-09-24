import json
st = json.load(open('/mnt/storage_pool/liaoyuanjun/pub_runs/pub_scheduler_state.json'))
print("status:", st.get("status"))
print("done:", len(st["done"]), "failed:", len(st["failed"]), "active:", len(st["active"]))
print("\nactive:")
for g, i in st["active"].items():
    print(f"  gpu={g} {i['rid']} pid={i['pid']}")
print("\nb0 done records:")
for d in st["done"]:
    if d["rid"].startswith("b0"):
        print(f"  {d['rid']:12s} contaminated={d.get('contaminated')} note={d.get('note','')}")
import os
PUB = "/mnt/storage_pool/liaoyuanjun/pub_runs"
for r in ("b0_drjohnson", "b0_flowers", "b0_counter", "b0_counter_contaminated_attempt1"):
    p = os.path.join(PUB, r, "results.json")
    print(f"{r}: results.json {'EXISTS' if os.path.exists(p) else 'missing'}")
# contamination flags for the in-flight runs
print("\ncontamination flags:", {k: v for k, v in st.get("contaminated", {}).items()})

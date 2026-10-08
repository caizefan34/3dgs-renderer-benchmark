import json,glob,os
base=r"C:\Users\36570\3dgs-renderer-benchmark\cgt_paper13\results"
for fn in ["PAPER13_PREREG.json","PAPER13_METHOD_FREEZE.json","PAPER13_SCENE_REGISTRY.json","PAPER13_SOURCE_AUDIT.json"]:
    p=os.path.join(base,fn)
    try:
        d=json.load(open(p,encoding="utf-8"))
        print("OK  ",fn,"| top keys:",list(d.keys())[:8])
    except Exception as e:
        print("FAIL",fn,e)
d=json.load(open(os.path.join(base,"PAPER13_SOURCE_AUDIT.json"),encoding="utf-8"))
gm=[d["per_scene"][s]["provenance"]["reference_v1_gm_hash_match"] for s in d["per_scene"]]
print("\nAll 13 gm_match True:",all(gm),"| count:",sum(gm))
print("F0 ingestible:",d["summary"]["f0_ingestible_count"],"/13 | training_required:",d["summary"]["training_required"])
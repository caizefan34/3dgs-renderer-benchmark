import json
d=json.load(open(r"C:\Users\36570\3dgs-renderer-benchmark\cgt_paper13\results\PAPER13_SOURCE_AUDIT.json"))
print("PARSE OK, scenes:",len(d["per_scene"]))
for s,v in d["per_scene"].items():
    print(f"{s:10s} {v['classification']:22s} N={v['gaussian_count']:8d} sha={v['checkpoint_sha256'][:12]} it={v['checkpoint_iteration']} absgrad={v['provenance']['absgrad']} gm_match={v['provenance']['reference_v1_gm_hash_match']}")
import json,sys;d=json.load(open(sys.argv[1]))
for k in ["top_1pct_share","top_5pct_share","top_10pct_share","p99","max"]:
    print(k,d["cross_camera_aggregate"][k])

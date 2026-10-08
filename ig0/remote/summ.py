import json, sys
path = sys.argv[1]
d = json.load(open(path))
keys = ("valid_source_descent_count","R_src_median","R_src_p10","peer_win_rate",
        "truehold_win_rate","truehold_psnr_delta_median_db","truehold_psnr_delta_worst_db",
        "correctness_pass","nan_inf_count")
for c, m in d["cells"].items():
    print("==== %s ====" % c)
    for v, mm in m["variants"].items():
        row = {k: (round(mm[k], 4) if isinstance(mm[k], float) else mm[k]) for k in keys}
        print(" ", v, row)

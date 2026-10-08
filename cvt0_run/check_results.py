import json
d = json.load(open("/root/epic05_cvt0/CVT0_RESULTS.json"))
print("valid json, cells:", list(d.keys()))
p0 = d["room@10000"]["panels"]["panel_0"]
print("has base_panel_eval:", "base_panel_eval" in p0)
print("has post_holdout_eval:", "post_holdout_eval" in p0)
print("base_panel_eval[0]:", p0["base_panel_eval"][0])

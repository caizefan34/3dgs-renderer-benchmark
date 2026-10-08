import json
d = json.load(open("/root/mr0_out_test/room@10000_groups.json"))
print("N=", d["N"], "n_groups=", d["n_groups"], "sh=", d["sh"])
print("isect_cams=", d["isect_cams"])
print("train_eval=", d["train_eval"])
print("fit_pool_len=", len(d["fit_pool"]), "first10=", d["fit_pool"][:10])
print("hold_eval=", d["hold_eval"])

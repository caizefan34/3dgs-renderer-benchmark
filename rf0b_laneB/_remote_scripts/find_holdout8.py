import json, os, numpy as np
cands = ["/root/sb0_fgs/holdout8", "/root/epic05_p12", "/root/epic05_p12/runs"]
for c in cands:
    if os.path.isdir(c):
        print("== DIR", c, "==")
        for name in sorted(os.listdir(c)):
            p=os.path.join(c,name)
            print("  ", "DIR " if os.path.isdir(p) else "FILE", name)
# deeper scan for holdout8 run dirs with camera_sequence.npy
print("==== scan for camera_sequence.npy under candidate roots ====")
for root in ["/root/sb0_fgs","/root/epic05_p12"]:
    for dp, dns, fns in os.walk(root):
        if "camera_sequence.npy" in fns:
            print(os.path.join(dp,"camera_sequence.npy"))

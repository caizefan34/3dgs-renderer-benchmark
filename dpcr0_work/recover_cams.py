import sys
sys.path.insert(0, "/root/epic05_n1")
sys.path.insert(0, "/root/3dgs-renderer-benchmark/scripts/epic05/phase7")
sys.path.insert(0, "/root/3dgs-renderer-benchmark/src")
sys.path.insert(0, "/root/3dgs-renderer-benchmark/baseline/reference_v1")
import n1_core as C
import numpy as np
for scene in ("room","bicycle"):
    ds = C.load_scene_dataset(scene)
    n = len(ds)
    train, holdout = [i for i in range(n) if i%8!=0], [i for i in range(n) if i%8==0]
    srcs=[]
    for s in range(8):
        lo=s*len(train)//8; hi=(s+1)*len(train)//8
        if hi<=lo: hi=min(lo+1,len(train))
        srcs.append(train[(lo+hi)//2])
    print(scene, "n=",n, "train_n=",len(train), "sources=",srcs)
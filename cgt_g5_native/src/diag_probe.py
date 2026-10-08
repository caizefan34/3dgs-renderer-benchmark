#!/usr/bin/env python
"""diag_probe.py - probe workload structure for geometric (ti,tj) recovery."""
import numpy as np
W="/home/liaoyuanjun/cgt_f0_fused_cuda/results/workload.npz"
d=np.load(W)
m2=d["means2d"]; off=d["tile_off"]; sc=d["scene"]; vw=d["view"]
print("means2d",m2.shape,"min",m2.min(0),"max",m2.max(0))
print("tile_off",off.shape,"first5",off[:5],"last3",off[-3:])
mono=bool(np.all(np.diff(off)>=0)); print("off monotonic nondecreasing:",mono)
print("off[0]==0:",int(off[0])==0,"off[-1]==len(m2):",int(off[-1])==len(m2))
cnt=np.diff(off); print("per-tile gauss count: min",cnt.min(),"median",int(np.median(cnt)),"max",cnt.max())
# per-tile centroid -> (ti,tj) via floor(centroid/16)
print("--- geometric (ti,tj) from centroid, first 8 tiles per scene ---")
for scene in ["room","garden","bicycle"]:
    idx=np.where(sc==scene)[0]
    for t in idx[:8]:
        s,e=int(off[t]),int(off[t+1]); g=m2[s:e]
        cx,cy=g[:,0].mean(),g[:,1].mean()
        ti=int(cy//16); tj=int(cx//16)
        print("%s t=%3d v=%3d n=%4d centroid=(%.1f,%.1f) -> ti=%d tj=%d  cxrange[%.1f,%.1f] cyrange[%.1f,%.1f]"%(
            scene,t,vw[t],e-s,cx,cy,ti,tj,g[:,0].min(),g[:,0].max(),g[:,1].min(),g[:,1].max()))

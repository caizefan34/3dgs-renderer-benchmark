#!/usr/bin/env python3
"""
Frozen A0-v2 structural analyzer.

Input .npz schema per shard/bucket:
  q_alpha: [256, m] float32
  q_color: [256, m, 3] float32
Optional metadata arrays are ignored by structure analysis.

This script performs ONLY structural SVD/DCT analysis.
VJP fidelity is computed by the execution-side exact replay because it depends
on the frozen kernel's exact local derivative factorization.
"""
import argparse, json
from pathlib import Path
import numpy as np

RANKS=[1,2,4,8,16,32]

def dct_basis_1d(n=16):
    B=np.zeros((n,n),dtype=np.float64)
    for k in range(n):
        a=np.sqrt(1/n) if k==0 else np.sqrt(2/n)
        for x in range(n):
            B[x,k]=a*np.cos(np.pi*(x+0.5)*k/n)
    return B

def dct_basis_2d():
    B1=dct_basis_1d(16)
    cols=[]
    idx=[]
    for fy in range(16):
        for fx in range(16):
            # x-fast flattening consistent with np.reshape(16,16).ravel C order.
            img=np.outer(B1[:,fy],B1[:,fx])
            cols.append(img.reshape(-1))
            idx.append((fx,fy))
    order=sorted(range(256),key=lambda i:(sum(idx[i]),idx[i][0],idx[i][1]))
    B=np.stack([cols[i] for i in order],axis=1)
    # Numerical check only.
    err=np.max(np.abs(B.T@B-np.eye(256)))
    if err>1e-10: raise RuntimeError(f"DCT basis not orthonormal: {err}")
    return B

B=dct_basis_2d()

def rank_stats(Q):
    Q=np.asarray(Q,dtype=np.float64)
    e=float(np.sum(Q*Q))
    if e==0:
        return {
            "energy":0.0,"effective_rank":0.0,
            "oracle_retention":{str(r):1.0 for r in RANKS},
            "dct_retention":{str(r):1.0 for r in RANKS},
            "r90":0,"r95":0,"r99":0,"r995":0
        }
    s=np.linalg.svd(Q,compute_uv=False)
    se=s*s
    cum=np.cumsum(se)/np.sum(se)
    def req(t):
        return int(np.searchsorted(cum,t)+1)
    eff=float((np.sum(se)**2)/np.sum(se**2))
    oracle={str(r):float(cum[min(r,len(cum))-1]) for r in RANKS}
    dct={}
    for r in RANKS:
        Br=B[:,:r]
        Qh=Br@(Br.T@Q)
        dct[str(r)]=float(np.sum(Qh*Qh)/e)
    return {
        "energy":e,"effective_rank":eff,
        "oracle_retention":oracle,
        "dct_retention":dct,
        "r90":req(.90),"r95":req(.95),"r99":req(.99),"r995":req(.995)
    }

ap=argparse.ArgumentParser()
ap.add_argument("--npz",nargs="+",required=True)
ap.add_argument("--out",required=True)
a=ap.parse_args()

rows=[]
for f in a.npz:
    d=np.load(f,allow_pickle=False)
    qa=d["q_alpha"]
    qc=d["q_color"]
    if qa.ndim!=2 or qa.shape[0]!=256:
        raise SystemExit(f"{f}: q_alpha shape {qa.shape}")
    if qc.ndim!=3 or qc.shape[0]!=256 or qc.shape[2]!=3 or qc.shape[1]!=qa.shape[1]:
        raise SystemExit(f"{f}: q_color shape {qc.shape}")
    qjoint=qc.transpose(0,2,1).reshape(256,-1)
    rows.append({
        "file":str(f),
        "m":int(qa.shape[1]),
        "alpha":rank_stats(qa),
        "color_joint":rank_stats(qjoint)
    })

Path(a.out).write_text(json.dumps({"rows":rows},indent=2),encoding="utf-8")
print(json.dumps({"n":len(rows),"out":a.out}))

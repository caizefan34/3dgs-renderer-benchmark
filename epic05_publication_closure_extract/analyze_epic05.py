#!/usr/bin/env python3
"""Aggregate EPIC-05 publication closure results without modifying frozen original artifacts."""
import argparse,json,math,pathlib,statistics
ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); a=ap.parse_args()
R=pathlib.Path(a.root)
scenes=["bicycle","bonsai","counter","drjohnson","flowers","garden","kitchen","playroom","room","stump","train","treehill","truck"]
def get(rid):
    d=json.load(open(R/rid/"results.json"))
    return {"wall":d["timing"]["total_wall_s"],"psnr":d["final_eval"]["psnr"],
            "ssim":d["final_eval"].get("ssim"),"lpips":d["final_eval"].get("lpips")}
def gm(v): return math.exp(sum(math.log(x) for x in v)/len(v))
out={}
# P1
rat=[]; dq=[]; per={}
for s in scenes:
    b=get(f"b1a_s42_{s}"); c=get(f"c0_s42_{s}")
    sp=b["wall"]/c["wall"]; rat.append(sp); dq.append(c["psnr"]-b["psnr"])
    per[s]={"speedup":sp,"dPSNR":c["psnr"]-b["psnr"]}
out["P1"]={"geomean_speedup":gm(rat),"n_faster":sum(x>1 for x in rat),
           "mean_dPSNR":sum(dq)/len(dq),"per_scene":per}
# P2 cumulative ladder
steps=[("A0_to_A1","a0","a1"),("A1_to_A2","a1","a2"),("A2_to_C0","a2","c0"),("A0_to_C0","a0","c0")]
out["P2"]={}
for name,x,y in steps:
    rr=[]; qq=[]
    for s in scenes:
        a0=get(f"{x}_s42_{s}") if x!="c0" else get(f"c0_s42_{s}")
        a1=get(f"{y}_s42_{s}") if y!="c0" else get(f"c0_s42_{s}")
        rr.append(a0["wall"]/a1["wall"]); qq.append(a1["psnr"]-a0["psnr"])
    out["P2"][name]={"geomean_speedup":gm(rr),"n_faster":sum(z>1 for z in rr),"mean_dPSNR":sum(qq)/len(qq)}
# P3 eps2d 2x2
out["P3"]={}
for s in ("room","bicycle","garden"):
    corners={
      "b1a_e01":get(f"b1a_s42_{s}"),
      "b1a_e03":get(f"b1ae03_s42_{s}"),
      "c0_e01":get(f"c0e01_s42_{s}"),
      "c0_e03":get(f"c0_s42_{s}")}
    out["P3"][s]=corners
# P4 multiseed
out["P4"]={}
for seed in (42,43,44):
    rr=[]; qq=[]
    for s in ("drjohnson","train","bicycle","room"):
        b=get(f"b1a_s{seed}_{s}"); c=get(f"c0_s{seed}_{s}")
        rr.append(b["wall"]/c["wall"]); qq.append(c["psnr"]-b["psnr"])
    out["P4"][str(seed)]={"geomean_speedup":gm(rr),"all_faster":all(z>1 for z in rr),"mean_dPSNR":sum(qq)/len(qq)}
json.dump(out,open(R.parent/"epic05_aggregate.json","w"),indent=2)
print(json.dumps(out,indent=2))

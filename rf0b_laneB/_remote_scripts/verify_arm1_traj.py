import json, os
root="/root/sb0_fgs/holdout8"
for sc in ["room","garden","bicycle"]:
    d=os.path.join(root,sc)
    rj=os.path.join(d,"results.json")
    print("==",sc,"==")
    if os.path.exists(rj):
        r=json.load(open(rj))
        print("results keys:", list(r.keys()))
        pi=r.get("per_iter",[])
        print("per_iter len:", len(pi), "sample:", pi[0] if pi else None, pi[-1] if pi else None)
        de=r.get("densification_events",[])
        print("dens_events:", len(de), "sample keys:", list(de[0].keys()) if de else None)
    mj=os.path.join(d,"memory.json")
    if os.path.exists(mj): print("memory.json:", json.load(open(mj)))
    # training_curve.csv head
    cc=os.path.join(d,"training_curve.csv")
    if os.path.exists(cc):
        lines=open(cc).read().splitlines()
        print("curve lines:", len(lines), "head:", lines[0], "| n_gs@30000:", lines[-1].split(",")[6] if len(lines)>1 else None)
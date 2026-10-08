#!/usr/bin/env python3
"""Compare the frozen four-view native and isolated-extension capture files."""
import argparse
import json
from pathlib import Path
import torch

CAMERAS=("DSCF4668","DSCF4770","DSCF4873","DSCF4976")
PARAMS=("_xyz","_features_dc","_features_rest","_scaling","_rotation","_opacity")

def metrics(ref, got):
    d=(got.float()-ref.float()).reshape(-1)
    r=ref.float().reshape(-1)
    max_abs=float(d.abs().max()) if d.numel() else 0.0
    rel=float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(r).clamp_min(1e-30)) if d.numel() else 0.0
    return {"max_abs":max_abs,"relative_l2":rel,"pass":rel<=1e-3 or max_abs<=1e-7}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("native",type=Path); ap.add_argument("candidate",type=Path); ap.add_argument("--output",type=Path,required=True); args=ap.parse_args()
    n=torch.load(args.native,map_location="cpu",weights_only=False)
    c=torch.load(args.candidate,map_location="cpu",weights_only=False)
    assert n["camera_names"]==c["camera_names"]==CAMERAS
    rows=[]; ok=True
    for a,b in zip(n["rows"],c["rows"]):
        assert a["camera"]==b["camera"]
        image=(a["image"].float()-b["image"].float()).abs()
        row={"camera":a["camera"],"rgb_max_abs":float(image.max()),
             "loss_abs":abs(float(a["loss"])-float(b["loss"])),
             "parameters":{k:metrics(a["gradients"][k],b["gradients"][k]) for k in PARAMS},
             "screen_all_four":metrics(a["screen_grad"],b["screen_grad"]),
             "screen_signed_xy":metrics(a["screen_grad"][:,:2],b["screen_grad"][:,:2]),
             "mean_absgrad_xy":metrics(a["screen_grad"][:,2:4],b["screen_grad"][:,2:4])}
        row["pass"]=(row["rgb_max_abs"]<=1e-4 and row["loss_abs"]<=1e-6 and
                     all(x["pass"] for x in row["parameters"].values()) and
                     row["screen_signed_xy"]["pass"] and row["mean_absgrad_xy"]["pass"])
        ok &= row["pass"]; rows.append(row)
    result={"cameras":CAMERAS,"rgb_max_abs_tolerance":1e-4,"loss_abs_tolerance":1e-6,
            "gradient_rule":"per group rel-L2 <= 1e-3 OR max-abs <= 1e-7",
            "source_commit":n["source_commit"],"ply_sha256":n["ply_sha256"],
            "native_module":n["module_file"],"candidate_module":c["module_file"],
            "all_pass":ok,"rows":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"all_pass":ok,"output":str(args.output),"per_view":[
        {"camera":r["camera"],"rgb":r["rgb_max_abs"],"loss":r["loss_abs"],"screen_rel":r["screen_all_four"]["relative_l2"],"absgrad_rel":r["mean_absgrad_xy"]["relative_l2"],"pass":r["pass"]} for r in rows]},indent=2))

if __name__=="__main__": main()

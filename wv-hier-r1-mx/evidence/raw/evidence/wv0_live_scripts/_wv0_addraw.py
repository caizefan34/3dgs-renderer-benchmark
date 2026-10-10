#!/usr/bin/env python3
"""Add WV0_GRADIENT_RAW.json (per-mask raw metric records) + gpu_snapshots, regen MANIFEST."""
import sys, os, json, glob, hashlib, torch, time
OUT = sys.argv[1] if len(sys.argv)>1 else "/dev/shm/wv0_out"
DEST = os.path.join(OUT, "WV_HIER_ORACLE0_MX")
GRAN=[16,8,4,1]; ALL=["means","quats","scales","opacities","colors"]

# The per-state .pt stores per-mask metrics. We emit them as the raw gradient record.
states={}
for f in sorted(glob.glob(os.path.join(OUT,"states","*.pt"))):
    s=torch.load(f, weights_only=False)
    sid=s["state_id"]
    rec={"state_id":sid,"scene":s["scene"],"iteration":s["iteration"],
         "N":s["N"],"n_cams":s["n_cams"],"resolution":s["cam_hw"],
         "ckpt_sha256":s["ckpt_sha256"],"conditions":{}}
    for ck,c in s["conditions"].items():
        cond={"sel_px_mean":c["sel_px_mean"],"sel_tiles_mean":c["sel_tiles_mean"],
              "total_pixels":c["total_pixels"],"groups":{}}
        for g in ALL:
            gv=c["groups"][g]
            cond["groups"][g]={
                "rel_l2_per_mask": gv["rel_l2"],
                "cos_per_mask": gv["cos"],
                "nmse_per_mask": gv["nmse"],
                "variance": gv["variance"],
                "bias_rel_l2": gv["bias_rel_l2"],
                "bias_cos": gv["bias_cos"],
            }
        rec["conditions"][ck]=cond
    states[sid]=rec

json.dump({"schema":"WV0_GRADIENT_RAW.v1",
           "description":"per-mask raw gradient metric records (32 masks, seeds 0..31) for each (state, condition, group). Full per-mask gradient tensors were not persisted (would be multi-GB); the per-mask metric arrays are the exact raw statistical record.",
           "n_masks":32,"mask_seeds":"0..31","states":states},
          open(os.path.join(DEST,"WV0_GRADIENT_RAW.json"),"w"), indent=2)
print("WV0_GRADIENT_RAW.json written, states:", len(states))

# gpu_snapshots: record GPU env at campaign time
gpu_snap={
  "host":"bms-39468022-001","gpus_used":[4,5,6,7],
  "gpu_model":"NVIDIA A100-PCIE-40GB",
  "cuda":"12.4 (conda nvcc) + driver 595.71.05",
  "torch":"2.4.1+cu124",
  "env":"conda anyplat (anysplat)",
  "timestamp":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "note":"structural-work oracle; no measured GPU runtime claimed for G8/G4"
}
json.dump(gpu_snap, open(os.path.join(DEST,"gpu_snapshots","gpu_env.json"),"w"), indent=2)
# per-state GPU snapshot (peak mem approx from work counters)
snap={"description":"per-state structural workload snapshot (from frozen projected state)",
      "states":{}}
for sid in states:
    s=torch.load(os.path.join(OUT,"states",sid+".pt"), weights_only=False)
    snap["states"][sid]={"N":s["N"],"visible_gaussians":s["work"]["visible_gaussians"],
                         "dense_isects_G16":s["work"]["per_G"]["16"]["dense_isects"]}
json.dump(snap, open(os.path.join(DEST,"gpu_snapshots","work_snapshot.json"),"w"), indent=2)
print("gpu_snapshots written")

# regenerate MANIFEST
lines=[]
for root,_,files in os.walk(DEST):
    for fn in sorted(files):
        if fn=="MANIFEST.sha256": continue
        p=os.path.join(root,fn)
        h=hashlib.sha256(open(p,"rb").read()).hexdigest()
        rel=os.path.relpath(p,DEST).replace("\\","/")
        lines.append(f"{h}  {rel}")
lines.sort()
open(os.path.join(DEST,"MANIFEST.sha256"),"w").write("\n".join(lines)+"\n")
print("MANIFEST regenerated:", len(lines), "files")

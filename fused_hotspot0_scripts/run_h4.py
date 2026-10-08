#!/usr/bin/env python3
"""FUSED-HOTSPOT0 H4 stats driver: workload stats across 3x3 via subprocess."""
import subprocess, os, sys, json, time

P = "/mnt/storage_pool/liaoyuanjun/fused_profile0"
PY = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python3.10"
HOT = P + "/FUSED_HOTSPOT0"
RAW = HOT + "/raw"
STAGE_CKPT = {"s1_5k":"0005000.pt","s2_10k":"0010000.pt","s3_mature":"final.pt"}
CKPT_ROOT = {
    "room":   "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_room_2026-10-06-01-01-33/checkpoints",
    "bicycle":"/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints",
    "garden": "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_garden_2026-10-06-01-32-48/checkpoints",
}
ENV = dict(os.environ)
ENV["CUDA_VISIBLE_DEVICES"]="0"; ENV["TMPDIR"]="/dev/shm"; ENV["TEMP"]="/dev/shm"; ENV["TMP"]="/dev/shm"
ENV["PYTHONPATH"] = ":".join([P+"/pylibs", P+"/methods_overlay",
    "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/src",
    "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/scripts"])

def run(scene, stage):
    out = f"{RAW}/stats_{scene}_{stage}.json"
    cmd = [PY, HOT+"/scripts/hotspot.py", "--config", f"{P}/profile/mat_{scene}.yaml",
           "--checkpoint", os.path.join(CKPT_ROOT[scene], STAGE_CKPT[stage]),
           "--scene", scene, "--stage", stage, "--mode","stats","--out", out]
    r = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
    ok = r.returncode==0 and os.path.exists(out)
    line = {"scene":scene,"stage":stage,"rc":r.returncode,"ok":ok}
    if not ok: line["stderr"]=r.stderr[-1200:]
    else:
        j=json.load(open(out)); line["n_instances"]=j["n_instances_tile_intersections"]; line["n_buckets"]=j["n_buckets"]; line["nG"]=j["n_gaussians_total"]
    print("STATSDONE", json.dumps(line), flush=True)
    return line

if __name__=="__main__":
    res=[]
    for scene in ["room","bicycle","garden"]:
        for stage in ["s1_5k","s2_10k","s3_mature"]:
            res.append(run(scene,stage))
    json.dump({"schema":"FUSED_HOTSPOT0_H4_DRIVER","runs":res}, open(RAW+"/h4_driver.json","w"), indent=2)
    print("H4_ALL_DONE", flush=True)
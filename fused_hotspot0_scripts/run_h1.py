#!/usr/bin/env python3
"""FUSED-HOTSPOT0 H1 driver: clean-GPU guard then clean timing across 3x3."""
import subprocess, os, sys, json, time, statistics

P = "/mnt/storage_pool/liaoyuanjun/fused_profile0"
PY = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python3.10"
HOT = P + "/FUSED_HOTSPOT0"
RAW = HOT + "/raw"
os.makedirs(RAW, exist_ok=True)

STAGE_CKPT = {
    "s1_5k": "0005000.pt",
    "s2_10k": "0010000.pt",
    "s3_mature": "final.pt",
}
SCENES = {
    "room": P + "/../strong_baselines/nerficg/output/FasterGSFused/fp1_room_2026-10-06-01-01-33/checkpoints",
    "bicycle": P + "/../strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints",
    "garden": P + "/../strong_baselines/nerficg/output/FasterGSFused/fp1_garden_2026-10-06-01-32-48/checkpoints",
}
SCENES = {k: os.path.normpath(v) for k, v in SCENES.items()}

ENV = dict(os.environ)
ENV["CUDA_VISIBLE_DEVICES"] = "0"
ENV["TMPDIR"] = "/dev/shm"; ENV["TEMP"] = "/dev/shm"; ENV["TMP"] = "/dev/shm"
ENV["PYTHONPATH"] = ":".join([P+"/pylibs", P+"/methods_overlay",
    "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/src",
    "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/scripts"])

def nvidia():
    return subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True).strip().splitlines()

def guard():
    # observe GPU 0 for >=20s; require mem<=512MiB and util<=2%
    samples = []
    t0 = time.time()
    while time.time() - t0 < 21:
        line = nvidia()[0].split(",")
        mem = int(line[1].strip()); util = int(line[2].strip())
        samples.append((mem, util)); time.sleep(2)
    mx_mem = max(m for m,_ in samples); mx_util = max(u for _,u in samples)
    ok = (mx_mem <= 512) and (mx_util <= 2)
    rec = {"schema":"FUSED_HOTSPOT0_GUARD","gpu":"0","observation_sec":round(time.time()-t0,1),
           "n_samples":len(samples),"max_mem_observed_mib":mx_mem,"max_util_observed_pct":mx_util,
           "clean":ok,"raw_samples":samples}
    json.dump(rec, open(RAW+"/guard_gpu0.json","w"), indent=2)
    return ok, rec

def run(scene, stage):
    ck = os.path.join(SCENES[scene], STAGE_CKPT[stage])
    out = f"{RAW}/clean_{scene}_{stage}.json"
    cmd = [PY, HOT+"/scripts/hotspot.py", "--config", f"{P}/profile/mat_{scene}.yaml",
           "--checkpoint", ck, "--scene", scene, "--stage", stage, "--mode","clean",
           "--warmup","50","--iters","200","--out", out]
    t0 = time.time()
    r = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
    dt = time.time()-t0
    ok = r.returncode == 0 and os.path.exists(out)
    line = {"scene":scene,"stage":stage,"rc":r.returncode,"wall_sec":round(dt,1),"ok":ok}
    if ok:
        j = json.load(open(out))
        line["median_total_ms"] = j["median_total_ms"]; line["nG"] = j["n_gaussians"]
        line["vram_gb"] = round(j["peak_vram_bytes"]/1e9,2)
    else:
        line["stderr"] = r.stderr[-1500:]
    print("H1DONE", json.dumps(line), flush=True)
    return line

if __name__ == "__main__":
    print("=== GUARD ===", flush=True)
    ok, rec = guard()
    print("GUARD", json.dumps({k:rec[k] for k in ["clean","max_mem_observed_mib","max_util_observed_pct"]}), flush=True)
    if not ok:
        print("GUARD_FAIL: GPU not clean; aborting H1", flush=True); sys.exit(2)
    results = []
    for scene in ["room","bicycle","garden"]:
        for stage in ["s1_5k","s2_10k","s3_mature"]:
            results.append(run(scene, stage))
    json.dump({"schema":"FUSED_HOTSPOT0_H1_DRIVER","guard":rec,"runs":results},
              open(RAW+"/h1_driver.json","w"), indent=2)
    print("H1_ALL_DONE", flush=True)
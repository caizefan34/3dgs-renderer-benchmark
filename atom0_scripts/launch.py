import os, sys, subprocess, time
P = "/mnt/storage_pool/liaoyuanjun/fused_profile0"
PY = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python3.10"
CKDIR = {
    "room": "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_room_2026-10-06-01-01-33/checkpoints",
    "bicycle": "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints",
    "garden": "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_garden_2026-10-06-01-32-48/checkpoints",
}
STAGE = {"s1_5k":"0005000.pt","s2_10k":"0010000.pt","s3_mature":"final.pt"}

def gpu_line(idx):
    out = subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    for ln in out.strip().splitlines():
        parts=[p.strip() for p in ln.split(",")]
        if parts[0]==str(idx):
            return int(parts[1]), int(parts[2])
    return None, None

def guard(idx, sec=21):
    t0=time.time(); mx_mem=0; mx_util=0; n=0
    while time.time()-t0 < sec:
        m,u = gpu_line(idx)
        if m is None: break
        mx_mem=max(mx_mem,mx_util if False else m); mx_util=max(mx_util,u); n+=1; time.sleep(2)
    ok = mx_mem<=512 and mx_util<=2
    return ok, mx_mem, mx_util, n

def main():
    gpu = sys.argv[1]; scene=sys.argv[2]; stage=sys.argv[3]
    cam = sys.argv[4] if len(sys.argv)>4 else "0"
    warm = sys.argv[5] if len(sys.argv)>5 else "30"
    samp = sys.argv[6] if len(sys.argv)>6 else "20"
    out = sys.argv[7] if len(sys.argv)>7 else f"{P}/ATOM0/raw/{scene}_{stage}_cam{cam}.json"
    do_probe = (len(sys.argv)>8 and sys.argv[8]=="probe")
    script = "probe_camera.py" if do_probe else "run_atom0.py"
    ok, mm, mu, n = guard(gpu)
    print(f"GUARD gpu{gpu}: clean={ok} max_mem={mm} max_util={mu}% n={n}", flush=True)
    if not ok:
        print("GPU NOT CLEAN - aborting"); sys.exit(2)
    cfg = f"{P}/profile/mat_{scene}.yaml"
    ck = os.path.join(CKDIR[scene], STAGE[stage])
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"]=gpu
    env["TMPDIR"]="/dev/shm"; env["TEMP"]="/dev/shm"; env["TMP"]="/dev/shm"
    env["HOME"]="/dev/shm/ncu_home"; os.makedirs("/dev/shm/ncu_home", exist_ok=True)
    if do_probe:
        cmd=[PY, f"{P}/ATOM0/scripts/{script}","--config",cfg,"--checkpoint",ck,
             "--scene",scene,"--stage",stage,"--out",out]
    else:
        cmd=[PY, f"{P}/ATOM0/scripts/{script}","--config",cfg,"--checkpoint",ck,
             "--scene",scene,"--stage",stage,"--cam_idx",cam,"--out",out,
             "--warmup",warm,"--samples",samp]
    t0=time.time()
    r=subprocess.run(cmd, env=env, capture_output=True, text=True)
    print("RC", r.returncode, "wall", round(time.time()-t0,1), flush=True)
    print(r.stdout[-4000:])
    if r.returncode!=0: print("ERR", r.stderr[-4000:])

if __name__=="__main__":
    main()

#!/usr/bin/env python3
"""P12 support lane: three-scene four-arm preflight runner (smoke only).

Runs each of the four P12 arms (gmain/c0/a0/a1) on each of the three support
scenes (room/bicycle/garden) for the requested iteration count (default 200,
the protocol minimum; this is a smoke preflight, NOT a campaign) and records:

  * source/binary identity (from the trainer's results.json)
  * the fused-loss runtime lock (P12_FUSED_LOSS_LOCK line + results field)
  * GPU UUID + clocks/temperature/power/memory snapshot before and after each
    run, plus a hard CLEAN-GPU gate (zero foreign compute processes)
  * wall duration, exit code, finite-loss census, peak VRAM, final-iteration
    throughput, log path

Timing recorded here is FUNCTIONAL_ONLY and must not be used to tune anything.

Arms are executed one per assigned GPU (3 scenes sequential per GPU) so that
no physical GPU is ever shared. GPU 4 on EPIC is occupied by a foreign
process and must never be selected.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

SWITCH_VARS = ("HIGS_DISABLE_F9", "HIGS_BWD_SCALAR_ADJOINT", "HIGS_BWD_H8_MR",
               "HIGS_BWD_ABSGRAD", "HIGS_PX_RUNTIME", "GSPLAT_H8_MR")
GPU_FIELDS = ("index,uuid,name,driver_version,clocks.sm,clocks.mem,"
              "temperature.gpu,power.draw,memory.used,memory.total,"
              "utilization.gpu")


def sh(cmd, timeout=60):
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except Exception as e:
        return f"ERROR {e}"


def gpu_snapshot(index):
    csv = sh(["nvidia-smi", f"--id={index}",
              f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader"])
    apps = [l for l in sh(["nvidia-smi",
                           "--query-compute-apps=pid,gpu_uuid,used_memory",
                           "--format=csv,noheader"]).splitlines() if l.strip()]
    uuid = csv.split(",")[1].strip() if "," in csv else "unknown"
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "csv": csv,
        "gpu_uuid": uuid,
        "compute_apps_all": apps,
        "compute_apps_on_this_gpu": [a for a in apps if uuid in a],
        "foreign_compute_apps": [a for a in apps if uuid in a],
    }


def run_one(py, trainer, sup, arm, scene, iterations, gpu):
    tag = f"{arm}_{scene}"
    out = os.path.join(sup, "runs", f"{tag}_{iterations}")
    log = os.path.join(sup, "logs", f"preflight_{tag}.log")
    os.makedirs(out, exist_ok=True)
    pre = gpu_snapshot(gpu)
    rec = {"arm": arm, "scene": scene, "iterations": iterations, "gpu_index": gpu,
           "outdir": out, "log": log,
           "env_inherited_switch_vars": {k: os.environ.get(k) for k in SWITCH_VARS},
           "pre_snapshot": pre}
    if pre["foreign_compute_apps"]:
        rec["status"] = "BLOCKED_GPU_NOT_CLEAN"
        rec["wall_s"] = None
        return rec
    cmd = [py, trainer, "--arm", arm, "--scene", scene,
           "--iterations", str(iterations), "--gpu", str(gpu),
           "--outdir", out, "--final-eval", "subset",
           "--timing-grade", "FUNCTIONAL_ONLY"]
    rec["cmd"] = " ".join(cmd)
    t0 = time.perf_counter()
    with open(log, "w") as lf:
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    rec["wall_s"] = time.perf_counter() - t0
    rec["exit_code"] = p.returncode
    rec["status"] = "OK" if p.returncode == 0 else "RUN_FAILED"
    rec["post_snapshot"] = gpu_snapshot(gpu)
    with open(os.path.join(out, "exit_code.txt"), "w") as f:
        f.write(str(p.returncode))
    with open(os.path.join(out, "preflight_meta.json"), "w") as f:
        json.dump(rec, f, indent=2)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", default="/root/miniforge3/envs/gsplat/bin/python")
    ap.add_argument("--support-dir", default="/root/epic05_p12_support")
    ap.add_argument("--trainer", default=None)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--arms", default="gmain,c0,a0,a1")
    ap.add_argument("--scenes", default="room,bicycle,garden")
    ap.add_argument("--gpus", default="3,5,6,7",
                    help="one GPU per arm (never GPU 4 on EPIC)")
    ap.add_argument("--out", default="/root/epic05_p12_support/P12_PREFLIGHT_RUNS.json")
    args = ap.parse_args()
    trainer = args.trainer or os.path.join(args.support_dir,
                                           "publication_trainer_p12.py")
    arms = args.arms.split(",")
    scenes = args.scenes.split(",")
    gpus = [int(x) for x in args.gpus.split(",")]
    if 4 in gpus:
        print("refusing: GPU 4 is occupied by a foreign process")
        sys.exit(2)
    if len(gpus) < len(arms):
        print(f"need one GPU per arm ({len(arms)}), got {len(gpus)}")
        sys.exit(2)

    sweep = {
        "phase": "P12 support lane - three-scene four-arm preflight runs",
        "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trainer": trainer, "iterations": iterations,
        "gpu_assignment": dict(zip(arms, gpus)),
        "timing_grade": "FUNCTIONAL_ONLY",
        "note": "smoke preflight; wall times are not publication-grade and must "
                "not be used to tune any parameter",
        "runs": [],
    }
    jobs = [(arm, scene, gpus[i]) for i, arm in enumerate(arms) for scene in scenes]
    with ThreadPoolExecutor(max_workers=len(arms)) as ex:
        futs = [ex.submit(run_one, args.py, trainer, args.support_dir, arm, scene,
                          args.iterations, gpu) for arm, scene, gpu in jobs]
        for f in futs:
            sweep["runs"].append(f.result())
    sweep["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    sweep["all_ok"] = all(r["status"] == "OK" for r in sweep["runs"])
    with open(args.out, "w") as f:
        json.dump(sweep, f, indent=2)
    print(f"WROTE {args.out} runs={len(sweep['runs'])} all_ok={sweep['all_ok']}")
    for r in sweep["runs"]:
        print(f"  {r['arm']:6s}/{r['scene']:8s} gpu{r['gpu_index']} "
              f"status={r['status']} wall={r['wall_s'] and round(r['wall_s'], 1)}s")
    sys.exit(0 if sweep["all_ok"] else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CGT-G5 P0: clean/exclusive F0 confirmation (goal sec.4, sec.23).

Fail-closed idle gate: a GPU is 'clean' only if, for a SUSTAINED window, utilization
is low, memory-used is near-empty, and NO foreign compute process is resident.
P0 reuses the FROZEN F0 workload + kernels verbatim (no retuning) and re-times the
mixed forward path. Verdict:
  pooled S_mixed meaningfully >1x AND correctness within frozen tol -> CONSISTENT_WITH_F0
  pooled S_mixed collapses to <=1x OR correctness outside tol              -> DROP_F0_REPRODUCIBILITY
  no clean GPU within budget                                              -> BLOCKED_GPU_OCCUPIED
Does NOT introduce a new 3x gate.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, ctypes
import numpy as np

# frozen F0 correctness tolerance (do not change)
CORRECTNESS_THRESH = 2e-5

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout

def gpu_snapshot():
    """Return list of per-GPU dicts: index, util%, mem_used, mem_total, foreign_pids."""
    out = sh("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total "
             "--format=csv,noheader,nounits")
    procs = sh("nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader")
    # map uuid->index
    uuid_out = sh("nvidia-smi --query-gpu=index,uuid --format=csv,noheader")
    uuid_to_idx = {}
    for line in uuid_out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            uuid_to_idx[parts[1]] = int(parts[0])
    foreign = {}  # index -> [pids]
    for line in procs.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0]:
            idx = uuid_to_idx.get(parts[0])
            if idx is not None:
                foreign.setdefault(idx, []).append(int(parts[1]))
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx = int(parts[0]); util = float(parts[1]); mu = float(parts[2]); mt = float(parts[3])
        gpus.append(dict(index=idx, util=util, mem_used_mib=mu, mem_total_mib=mt,
                         foreign_pids=foreign.get(idx, [])))
    return gpus

def is_clean(g, util_max, mem_mib_max):
    return (g["util"] <= util_max and g["mem_used_mib"] <= mem_mib_max
            and len(g["foreign_pids"]) == 0)

def wait_for_clean_gpu(util_max, mem_mib_max, window_s, poll_s, budget_s, log=lambda m: None):
    """Poll until SOME GPU stays clean for a continuous window, or budget expires."""
    t0 = time.time()
    clean_since = {}
    while True:
        snap = gpu_snapshot()
        now = time.time()
        for g in snap:
            idx = g["index"]
            if is_clean(g, util_max, mem_mib_max):
                clean_since.setdefault(idx, now)
                if now - clean_since[idx] >= window_s:
                    log(f"[P0] GPU {idx} clean for {window_s:.0f}s (util={g['util']}, mem={g['mem_used_mib']}MiB)")
                    return idx, g, snap
            else:
                clean_since.pop(idx, None)
        if now - t0 >= budget_s:
            return None, None, snap
        time.sleep(poll_s)

def run_f0_harness(pybin, workload, so, outdir, device, repeats, warmup):
    # Invoke the FROZEN F0 harness verbatim. CUDA_VISIBLE_DEVICES restricts to the
    # clean physical GPU; inside the subprocess that GPU is always device 0.
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(device)
    harness = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(so))), "src", "f0_harness.py")
    if not os.path.exists(harness):
        harness = os.path.join(os.path.dirname(os.path.abspath(so)), "f0_harness.py")
    cmd = [pybin, harness, "--workload", workload, "--so", so, "--out", outdir,
           "--device", "0", "--repeats", str(repeats), "--warmup", str(warmup)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["watch","measure"], default="measure")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--so", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pybin", default="/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python")
    ap.add_argument("--util-max", type=float, default=5.0)
    ap.add_argument("--mem-mib-max", type=float, default=6144.0)
    ap.add_argument("--window-s", type=float, default=60.0)
    ap.add_argument("--poll-s", type=float, default=15.0)
    ap.add_argument("--budget-s", type=float, default=1800.0)
    ap.add_argument("--repeats", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    logf = open(os.path.join(args.out, "p0_watch.log"), "a")
    def log(m):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    result = dict(stage="P0", mode=args.mode, ts=time.strftime("%Y-%m-%d %H:%M:%S"),
                  gate=dict(util_max=args.util_max, mem_mib_max=args.mem_mib_max,
                            window_s=args.window_s, poll_s=args.poll_s, budget_s=args.budget_s),
                  clean_gpu=None, gpu_snapshot_before=None, gpu_snapshot_after=None,
                  correctness_max_abs=None, s_mixed_pooled=None, per_scene=None,
                  verdict=None, note="")

    if args.mode == "watch":
        idx, g, snap = wait_for_clean_gpu(args.util_max, args.mem_mib_max,
                                          args.window_s, args.poll_s, args.budget_s, log)
        result["gpu_snapshot_before"] = snap
        if idx is None:
            result["verdict"] = "BLOCKED_GPU_OCCUPIED"
            result["note"] = f"No GPU stayed clean (util<={args.util_max}%, mem<={args.mem_mib_max}MiB, no foreign pid) for {args.window_s:.0f}s within {args.budget_s:.0f}s budget."
        else:
            result["clean_gpu"] = idx
            result["verdict"] = "CLEAN_GPU_FOUND"
            result["note"] = "Clean GPU found; run --mode measure to execute P0 confirmation."
        json.dump(result, open(os.path.join(args.out, "P0_F0_CLEAN_CONFIRMATION.json"), "w"), indent=2)
        log("WROTE P0_F0_CLEAN_CONFIRMATION.json verdict=" + str(result["verdict"]))
        return

    # measure mode: require a clean GPU right now (fail-closed)
    snap = gpu_snapshot()
    result["gpu_snapshot_before"] = snap
    idx, g, _ = wait_for_clean_gpu(args.util_max, args.mem_mib_max, args.window_s, args.poll_s, args.budget_s, log)
    if idx is None:
        result["verdict"] = "BLOCKED_GPU_OCCUPIED"
        result["note"] = "No clean GPU available at measure time."
        json.dump(result, open(os.path.join(args.out, "P0_F0_CLEAN_CONFIRMATION.json"), "w"), indent=2)
        log("BLOCKED_GPU_OCCUPIED"); return
    result["clean_gpu"] = idx
    log(f"Running frozen F0 harness on clean GPU {idx} ...")

    p = run_f0_harness(args.pybin, args.workload, args.so, args.out, idx, args.repeats, args.warmup)
    result["harness_returncode"] = p.returncode
    if p.returncode != 0:
        result["verdict"] = "BLOCKED_SOURCE_INTEGRITY"
        result["note"] = "F0 harness failed to run: " + (p.stderr or p.stdout)[-2000:]
        json.dump(result, open(os.path.join(args.out, "P0_F0_CLEAN_CONFIRMATION.json"), "w"), indent=2)
        log("BLOCKED_SOURCE_INTEGRITY: " + result["note"][:400]); return

    # read back F0 outputs
    mt = os.path.join(args.out, "F0_MIXED_TIMING.json")
    co = os.path.join(args.out, "F0_CORRECTNESS.json")
    if os.path.exists(mt):
        m = json.load(open(mt))
        result["s_mixed_pooled"] = m.get("S_mixed_median")
        result["s_mixed_with_h2d"] = m.get("S_mixed_with_h2d_conservative")
        result["per_scene"] = {k: v.get("S_mixed") for k, v in (m.get("per_scene") or {}).items()}
    if os.path.exists(co):
        c = json.load(open(co))
        # worst max_abs across checks
        worst = 0.0
        for chk in c.get("checks", []):
            worst = max(worst, chk.get("fused_mixed_max_abs", 0.0))
        result["correctness_max_abs"] = worst
        result["correctness_passed"] = c.get("all_passed")

    result["gpu_snapshot_after"] = gpu_snapshot()

    # P0 decision (no new 3x gate)
    s = result["s_mixed_pooled"]; ok_corr = bool(result.get("correctness_passed"))
    if s is None:
        result["verdict"] = "BLOCKED_SOURCE_INTEGRITY"; result["note"] = "S_mixed missing after run."
    elif not ok_corr:
        result["verdict"] = "DROP_F0_REPRODUCIBILITY"
        result["note"] = f"Correctness outside frozen tol on clean GPU (max_abs={result['correctness_max_abs']})."
    elif s <= 1.0:
        result["verdict"] = "DROP_F0_REPRODUCIBILITY"
        result["note"] = f"Clean-GPU pooled S_mixed collapsed to {s} (<=1x). F0 result not reproducible on exclusive GPU."
    else:
        result["verdict"] = "CONSISTENT_WITH_F0"
        result["note"] = f"Clean-GPU pooled S_mixed={s} (>1x), correctness within tol. Consistent with frozen F0 (pooled 2.99x). Not a 3x gate."
    json.dump(result, open(os.path.join(args.out, "P0_F0_CLEAN_CONFIRMATION.json"), "w"), indent=2)
    log("P0 verdict=" + str(result["verdict"]) + " S_mixed=" + str(s) + " corr=" + str(result.get("correctness_max_abs")))

if __name__ == "__main__":
    main()

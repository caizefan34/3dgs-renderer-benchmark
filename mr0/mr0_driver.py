#!/usr/bin/env python3
"""MR0 full-pipeline driver. For each cell: groups -> fit (sharded across GPUs)
-> replace. Then aggregate MR0_SUMMARY.json and run the FROZEN analyzer to
produce MR0_VERDICT.json. No production file is modified."""
import sys, os, json, subprocess, time, shutil

PKG = "/root/mr0_pkg"
OUT = "/root/mr0_out"
PY = "/root/miniforge3/envs/gsplat/bin/python"
GPUS = ["0", "1", "2", "4", "5", "6", "7"]
CELLS = ["room@10000", "room@15000", "bicycle@10000", "bicycle@15000"]
N_STEPS = int(os.environ.get("MR0_NSTEPS", "6"))
N_CAM_FIT = 8
N_ISECT = 128


def sh(cmd, gpu=None, log=None):
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    print("[driver] GPU=%s %s" % (gpu, cmd), flush=True)
    if log:
        f = open(log, "w")
        r = subprocess.run(cmd, shell=True, env=env, cwd=PKG, stdout=f, stderr=subprocess.STDOUT)
        f.close()
    else:
        r = subprocess.run(cmd, shell=True, env=env, cwd=PKG)
    return r.returncode


def main():
    os.makedirs(OUT, exist_ok=True)
    for cell in CELLS:
        t0 = time.perf_counter()
        rc = sh("%s mr0_cell.py groups --cell %s --out %s" % (PY, cell, OUT), GPUS[0],
                "%s/%s_groups.log" % (OUT, cell))
        if rc != 0:
            print("[driver] GROUPS FAILED %s rc=%d" % (cell, rc), flush=True); sys.exit(1)
        gj = json.load(open("%s/%s_groups.json" % (OUT, cell)))
        G = gj["n_groups"]
        nsh = len(GPUS)
        bounds = [i * G // nsh for i in range(nsh + 1)]
        procs = []
        for s in range(nsh):
            g0, g1 = bounds[s], bounds[s + 1]
            if g1 <= g0:
                continue
            log = "%s/%s_fit_%d.log" % (OUT, cell, s)
            env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = GPUS[s]
            print("[driver] fit shard %d: groups [%d,%d) GPU %s" % (s, g0, g1, GPUS[s]), flush=True)
            procs.append(subprocess.Popen(
                "%s mr0_cell.py fit --cell %s --out %s --g0 %d --g1 %d --n-steps %d --n-cam-fit %d"
                % (PY, cell, OUT, g0, g1, N_STEPS, N_CAM_FIT),
                shell=True, env=env, cwd=PKG, stdout=open(log, "w"), stderr=subprocess.STDOUT))
        for s, p in enumerate(procs):
            rc = p.wait()
            if rc != 0:
                print("[driver] FIT SHARD %d FAILED rc=%d" % (s, rc), flush=True); sys.exit(1)
        rc = sh("%s mr0_cell.py replace --cell %s --out %s --n-isect-sample %d" % (PY, cell, OUT, N_ISECT),
                GPUS[0], "%s/%s_replace.log" % (OUT, cell))
        if rc != 0:
            print("[driver] REPLACE FAILED %s rc=%d" % (cell, rc), flush=True); sys.exit(1)
        print("[driver] %s DONE in %.0fs" % (cell, time.perf_counter() - t0), flush=True)
    summary = {"requires_anchor_family_features": False, "cells": {}}
    for cell in CELLS:
        rep = json.load(open("%s/%s_replacement.json" % (OUT, cell)))
        summary["cells"][cell] = {"K": rep["K"], "scene": rep["scene"], "it": rep["it"],
                                  "N": rep["N"], "ckpt_sha256": rep["ckpt_sha256"],
                                  "n_groups": rep["n_groups"]}
    json.dump(summary, open("%s/MR0_SUMMARY.json" % OUT, "w"), indent=2)
    print("[driver] MR0_SUMMARY.json written", flush=True)
    shutil.copy("%s/mr0_analyze.py" % PKG, "%s/mr0_analyze.py" % OUT)
    rc = sh("%s %s/mr0_analyze.py %s/MR0_SUMMARY.json" % (PY, OUT, OUT), None, "%s/analyze.log" % OUT)
    print("[driver] analyze rc=%d" % rc, flush=True)
    rc = sh("%s %s/mr0_report.py %s" % (PY, PKG, OUT), None, "%s/report.log" % OUT)
    print("[driver] report rc=%d" % rc, flush=True)
    print("DRIVER_DONE", flush=True)


if __name__ == "__main__":
    main()

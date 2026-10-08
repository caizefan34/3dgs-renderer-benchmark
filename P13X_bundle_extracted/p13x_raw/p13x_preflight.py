#!/usr/bin/env python3
"""P13-X preflight (required before any long diagnostic run).

A. EQUIVALENCE: P12 trainer vs P13-X trainer (diagnostics OFF), 220 iters,
   room, same physical GPU back-to-back. Requires loss checkpoints
   1/100/200 to match within P12-style FP tolerances
   ({1: 1e-3, 100: 1e-2, 200: 2e-2} rel-diff, the P12 LOSS_SCALE_GATES
   convention) and renderer/binary hashes to match the frozen prereg.

B. CONTAMINATION: P13-X diagnostics ON (forced test window 200..300 so the
   non-F9 code path is exercised) vs diagnostics OFF, 300 iters, room,
   same GPU back-to-back, both with RNG footprint recording. Requires:
   - camera_sequence.npy byte-identical
   - per-iter losses at 1/100/200/300 within P12-style gates
   - RNG footprints identical at every milestone
   - n_gs trajectory identical
   - diagnostics actually ran (x4 samples, x5 probes, x1 pairs, snapshots)

NO TUNING OF ANY KIND. Output: /root/epic05_p13x/artifacts/P13X_PREFLIGHT.json
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

PY = "/root/miniforge3/envs/gsplat/bin/python"
P12 = "/root/epic05_p12/publication_trainer_p12.py"
P13X = "/root/epic05_p13x/publication_trainer_p13x.py"
OUT = "/root/epic05_p13x/preflight"
ART = "/root/epic05_p13x/artifacts"
SCENE = "room"
GPU = os.environ.get("P13X_PREFLIGHT_GPU", "2")
ITERS_A, ITERS_B = 220, 300
GATES_A = {1: 1e-3, 100: 1e-2, 200: 2e-2}          # P12 LOSS_SCALE_GATES
GATES_B = {1: 1e-3, 100: 1e-2, 200: 2e-2, 300: 2e-2}
PREREG_SO = "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11"
PREREG_CORE = "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e"
P11_FUSED = "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb"
RAW = ["P13X_RAW_X4.json", "P13X_RAW_X5.json", "P13X_RAW_X1.json",
       "P13X_RAW_SNAPSHOTS.json"]


def mk(script, iters, extra=()):
    return ([PY, script, "--arm", "c0", "--scene", SCENE, "--loss", "fused",
             "--gpu", "0", "--eps2d", "0.3", "--final-eval", "subset",
             "--timing-grade", "FUNCTIONAL_ONLY", "--lpips",
             "--iterations", str(iters)] + list(extra))


def run(cmd, outdir, gpu):
    outdir = pathlib.Path(outdir)
    if (outdir / "results.json").exists():
        return {"returncode": 0, "wall_s": 0.0, "outdir": str(outdir),
                "reused": True}
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = cmd + ["--outdir", str(outdir)]
    (outdir / "command.txt").write_text(" ".join(cmd) + "\n")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    t0 = time.time()
    with open(outdir / "stdout.log", "w", buffering=1) as f:
        p = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                           text=True)
    return {"returncode": p.returncode, "wall_s": time.time() - t0,
            "outdir": str(outdir)}


def losses_of(outdir):
    R = json.load(open(os.path.join(outdir, "results.json")))
    return {r["iter"]: r["loss"] for r in R["per_iter"]}, R


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    os.makedirs(ART, exist_ok=True)
    if os.path.exists(os.path.join(OUT, "lock")):
        print("PREFLIGHT ALREADY DONE (lock present); verifying only")
        return verify()
    pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
    report = {"scene": SCENE, "gpu": GPU,
              "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ")}

    # ---------------- Preflight A: P12 vs P13X (diag off) ----------------
    ra = run(mk(P12, ITERS_A), os.path.join(OUT, "A_p12_220"), GPU)
    rb = run(mk(P13X, ITERS_A, ["--p13x-diag", "0"]),
             os.path.join(OUT, "A_p13x_diagoff_220"), GPU)
    lA, RA = losses_of(os.path.join(OUT, "A_p12_220"))
    lB, RB = losses_of(os.path.join(OUT, "A_p13x_diagoff_220"))
    rows = {}
    okA = ra["returncode"] == 0 and rb["returncode"] == 0
    for it, gate in GATES_A.items():
        f, n = lA.get(it), lB.get(it)
        rel = abs(n - f) / abs(f) if (f and n) else None
        rows[str(it)] = {"p12": f, "p13x": n, "rel_diff": rel, "gate": gate}
        if rel is None or rel > gate:
            okA = False
    ident_checks = {
        "c0_so_sha256": RB["binary_identity"]["so_sha256"] == PREREG_SO,
        "c0_core_sha256":
            RB["binary_identity"].get("core_sha256") == PREREG_CORE,
        "p12_so_sha256": RA["binary_identity"]["so_sha256"] == PREREG_SO,
        "fused_so_sha256": RB["loss_backend"]["so_sha256"] == P11_FUSED,
        "loss_backend_fused": RB["loss_backend"]["name"] == "fused-ssim",
        "padding_same": RB["loss_backend"].get("padding") == "same",
    }
    for v in ident_checks.values():
        if not v:
            okA = False
    report["A_equivalence"] = {
        "runs": {"p12": ra, "p13x_diagoff": rb},
        "loss_checkpoints": rows,
        "identity_checks": ident_checks,
        "pass": bool(okA),
    }

    # ---------------- Preflight B: diag ON vs OFF ----------------
    rc = run(mk(P13X, ITERS_B, ["--p13x-diag", "0",
                                "--p13x-rng-footprint", "1"]),
             os.path.join(OUT, "B_diagoff_300"), GPU)
    rd = run(mk(P13X, ITERS_B,
                ["--p13x-diag", "1", "--p13x-diag-start", "200",
                 "--p13x-diag-end", "300", "--p13x-diag-stride", "50",
                 "--p13x-lags", "50,100", "--p13x-x4-samples", "20",
                 "--p13x-x5", "1", "--p13x-rng-footprint", "1"]),
             os.path.join(OUT, "B_diagon_300"), GPU)
    lC, RC = losses_of(os.path.join(OUT, "B_diagoff_300"))
    lD, RD = losses_of(os.path.join(OUT, "B_diagon_300"))
    rowsB = {}
    okB = rc["returncode"] == 0 and rd["returncode"] == 0
    for it, gate in GATES_B.items():
        f, n = lC.get(it), lD.get(it)
        rel = abs(n - f) / abs(f) if (f and n) else None
        rowsB[str(it)] = {"diagoff": f, "diagon": n, "rel_diff": rel,
                          "gate": gate}
        if rel is None or rel > gate:
            okB = False
    seq_off = os.path.join(OUT, "B_diagoff_300", "camera_sequence.npy")
    seq_on = os.path.join(OUT, "B_diagon_300", "camera_sequence.npy")
    seq_identical = (os.path.exists(seq_off) and os.path.exists(seq_on)
                     and sha256_file(seq_off) == sha256_file(seq_on))
    if not seq_identical:
        okB = False
    fp_off = [json.loads(x) for x in
              open(os.path.join(OUT, "B_diagoff_300",
                                "P13X_RNG_FOOTPRINT.jsonl")) if x.strip()]
    fp_on = [json.loads(x) for x in
             open(os.path.join(OUT, "B_diagon_300",
                               "P13X_RNG_FOOTPRINT.jsonl")) if x.strip()]
    rng_identical = (len(fp_off) == len(fp_on) and all(
        a["torch_cpu_rng_sha256"] == b["torch_cpu_rng_sha256"]
        and a["torch_cuda_rng_sha256"] == b["torch_cuda_rng_sha256"]
        for a, b in zip(fp_off, fp_on)))
    if not rng_identical:
        okB = False
    rawd = os.path.join(OUT, "B_diagon_300", "p13x", "raw")
    try:
        x4 = json.load(open(os.path.join(rawd, "P13X_RAW_X4.json")))
        x5 = json.load(open(os.path.join(rawd, "P13X_RAW_X5.json")))
        x1 = json.load(open(os.path.join(rawd, "P13X_RAW_X1.json")))
        snaps = json.load(open(os.path.join(rawd,
                                            "P13X_RAW_SNAPSHOTS.json")))
        diag_ran = {
            "x4_samples": len(x4), "x4_ok": len(x4) >= 20,
            "x5_probes": len(x5),
            "x5_errors": sum(1 for r in x5 if "error" in r),
            "x1_pairs": len(x1), "snapshots": len(snaps),
        }
        diag_ran["ok"] = (diag_ran["x4_ok"] and diag_ran["x5_probes"] >= 24
                          and diag_ran["x5_errors"] == 0
                          and diag_ran["x1_pairs"] > 0)
    except Exception as e:  # noqa: BLE001
        diag_ran = {"ok": False, "error": repr(e)}
    if not diag_ran.get("ok", False):
        okB = False
    ngs_off = {r["iter"]: r["n_gs"] for r in RC["per_iter"]}
    ngs_on = {r["iter"]: r["n_gs"] for r in RD["per_iter"]}
    ngs_equal = ngs_off == ngs_on
    if not ngs_equal:
        okB = False
    report["B_contamination"] = {
        "runs": {"diagoff": rc, "diagon": rd},
        "loss_checkpoints": rowsB,
        "camera_sequence_identical": seq_identical,
        "rng_footprints_identical": rng_identical,
        "rng_milestones": len(fp_on),
        "n_gs_trajectory_equal": ngs_equal,
        "diagnostics_ran": diag_ran,
        "pass": bool(okB),
    }
    report["overall_pass"] = bool(okA and okB)
    with open(os.path.join(ART, "P13X_PREFLIGHT.json"), "w") as f:
        json.dump(report, f, indent=2)
    pathlib.Path(os.path.join(OUT, "lock")).write_text("done\n")
    print(json.dumps(report, indent=2))
    return 0 if report["overall_pass"] else 1


def verify():
    r = json.load(open(os.path.join(ART, "P13X_PREFLIGHT.json")))
    print(json.dumps(r, indent=2))
    return 0 if r["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

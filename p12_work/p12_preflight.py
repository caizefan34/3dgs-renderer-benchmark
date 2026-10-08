#!/usr/bin/env python3
"""P12 preflight (required before any 30K run).

For every arm that will enter the campaign (GMAIN_FUSED, C0_FUSED, A0_FUSED,
A1_FUSED) on room / bicycle / garden:
  1. run the actual P12 trainer for 220 iterations (>= 200 required)
     with --loss fused;
  2. verify source SHA and .so SHA per arm;
  3. verify fused-SSIM source/binary hash (P11 lock) and padding="same"
     actually active (runtime probe result recorded by the trainer);
  4. confirm finite forward/loss/backward (all recorded losses finite,
     final eval finite, status SUCCESS);
  5. confirm loss scale comparable to the frozen recipe (compare iter 1/100/200
     losses against the frozen SepSSIM campaign runs of the same arm+scene);
  6. confirm expected switch behavior (gmain plain current main; c0 frozen C0;
     a0 F9 OFF; a1 F9 ON; no unexpected H8/Scalar env in a0/a1);
  7. record peak VRAM;
  8. record one short phase-timing sample.

NO TUNING OF ANY KIND. Output: /root/epic05_p12/artifacts/P12_PREFLIGHT.json
Overall PASS required before the 30K campaign; a semantic or source-identity
blocker STOPS P12.
"""
import json
import os
import subprocess
import sys
import threading
import time
import pathlib

PY = "/root/miniforge3/envs/gsplat/bin/python"
TRAINER = "/root/epic05_p12/publication_trainer_p12.py"
OUTROOT = "/root/epic05_p12/preflight"
ART = "/root/epic05_p12/artifacts"
FROZEN = "/root/3dgs-publication-epic05/full30k"
ITERATIONS = 220

SCENE_GPU = {"room": 5, "bicycle": 6, "garden": 7}
ARMS = ["gmain", "c0", "a0", "a1"]
FROZEN_RUN = {"gmain": "gmain03_s42_{scene}", "c0": "c0_s42_{scene}",
              "a0": "a0_s42_{scene}", "a1": "a1_s42_{scene}"}

EXPECTED_SO = {
    "gmain": "8ca8ec1e0c8618f058638b93010061a5bafb7f6c92005b54d07dc6c22658c811",
    "c0": "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11",
    "a0": "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11",
    "a1": "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11",
}
EXPECTED_CORE = {"c0": "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e",
                 "a0": "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e",
                 "a1": "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e"}
P11_FUSED_SO = "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb"

EXPECTED_ENV = {
    "gmain": {},
    "c0": {"HIGS_PX_RUNTIME": "2", "HIGS_BWD_SCALAR_ADJOINT": "scalar_adjoint",
           "HIGS_BWD_H8_MR": "1", "HIGS_BWD_ABSGRAD": "1"},
    "a0": {"HIGS_PX_RUNTIME": "2", "HIGS_BWD_ABSGRAD": "1",
           "HIGS_BWD_SCALAR_ADJOINT": None, "HIGS_BWD_H8_MR": None,
           "HIGS_DISABLE_F9": "1"},
    "a1": {"HIGS_PX_RUNTIME": "2", "HIGS_BWD_ABSGRAD": "1",
           "HIGS_BWD_SCALAR_ADJOINT": None, "HIGS_BWD_H8_MR": None,
           "HIGS_DISABLE_F9": None},
}

LOSS_SCALE_GATES = {1: 0.001, 100: 0.01, 200: 0.02}  # rel-diff gates per iter


def run_arm_scene(arm, scene, gpu):
    run_id = f"pre_{arm}_s42_{scene}"
    outdir = pathlib.Path(OUTROOT) / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, TRAINER, "--arm", arm, "--scene", scene,
           "--iterations", str(ITERATIONS), "--outdir", str(outdir),
           "--gpu", "0", "--lpips", "--final-eval", "subset",
           "--timing-grade", "FUNCTIONAL_ONLY", "--eps2d", "0.3",
           "--loss", "fused"]
    (outdir / "command.txt").write_text(" ".join(cmd) + "\n")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    t0 = time.time()
    with open(outdir / "stdout.log", "w", buffering=1) as f:
        p = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
    return {"run_id": run_id, "outdir": str(outdir), "returncode": p.returncode,
            "wall_s": time.time() - t0}


def verify(arm, scene, runinfo):
    checks = []
    rec = {"arm": arm, "arm_label": f"{arm.upper()}_FUSED", "scene": scene,
           "run_id": runinfo["run_id"], "returncode": runinfo["returncode"],
           "wall_s": round(runinfo["wall_s"], 1)}

    def add(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        return ok

    rp = os.path.join(runinfo["outdir"], "results.json")
    if not os.path.exists(rp):
        add("results_json_exists", False, rp)
        rec["checks"] = checks
        rec["ok"] = False
        return rec
    R = json.load(open(rp))
    FS = json.load(open(os.path.join(runinfo["outdir"], "final_status.json")))
    TJ = json.load(open(os.path.join(runinfo["outdir"], "timing.json")))
    MEM = json.load(open(os.path.join(runinfo["outdir"], "memory.json")))

    # 1/2. source & binary identity
    ident = R["binary_identity"]
    add("arm_so_sha256", ident["so_sha256"] == EXPECTED_SO[arm],
        {"got": ident["so_sha256"], "expected": EXPECTED_SO[arm], "path": ident.get("so_path")})
    if arm in EXPECTED_CORE:
        add("core_so_sha256", ident.get("core_sha256") == EXPECTED_CORE[arm],
            {"got": ident.get("core_sha256"), "expected": EXPECTED_CORE[arm]})
    add("arm_is_expected_arm", ident["arm"] == arm, ident["arm"])

    # 3. fused-SSIM identity + padding=same active
    LB = R["loss_backend"]
    add("loss_backend_is_fused", LB["name"] == "fused-ssim", LB["name"])
    add("fused_so_sha256_p11_lock", LB.get("so_sha256") == P11_FUSED_SO,
        {"got": LB.get("so_sha256"), "expected": P11_FUSED_SO})
    add("padding_is_same", LB.get("padding") == "same", LB.get("padding"))
    probe = LB.get("padding_same_probe", {})
    add("padding_same_probe_pass",
        probe.get("pass") is True and probe.get("same_shape") is True
        and probe.get("abs_diff", 1.0) <= probe.get("threshold", 0.0),
        probe)

    # 4. finite fwd/loss/bwd + >= 200 iters
    add("status_success", FS["status"] == "SUCCESS", FS["status"])
    add("iterations_ge_200", R["iterations"] >= 200, R["iterations"])
    losses = {r["iter"]: r["loss"] for r in R["per_iter"]}
    all_finite = all(v is not None and v == v and abs(v) < 1e9 for v in losses.values())
    add("all_recorded_losses_finite", all_finite,
        {k: losses[k] for k in sorted(losses)})
    fe = R["final_eval"]
    add("final_eval_finite",
        all(map(lambda x: x is not None and x == x and abs(x) < 1e9,
                [fe.get("psnr"), fe.get("ssim"), fe.get("l1")])),
        {"psnr": fe.get("psnr"), "ssim": fe.get("ssim"), "l1": fe.get("l1"),
         "lpips": fe.get("lpips")})

    # 5. loss scale comparable to frozen recipe
    frozen_dir = os.path.join(FROZEN, FROZEN_RUN[arm].format(scene=scene))
    scale_rows = {}
    scale_ok = True
    try:
        FR = json.load(open(os.path.join(frozen_dir, "results.json")))
        frozen_losses = {r["iter"]: r["loss"] for r in FR["per_iter"]}
        for it, gate in LOSS_SCALE_GATES.items():
            f = frozen_losses.get(it)
            n = losses.get(it)
            if f is None or n is None:
                scale_ok = False
                scale_rows[str(it)] = {"frozen": f, "p12": n, "rel_diff": None}
                continue
            rel = abs(n - f) / abs(f)
            scale_rows[str(it)] = {"frozen": f, "p12": n,
                                   "rel_diff": rel, "gate": gate}
            if rel > gate:
                scale_ok = False
    except Exception as e:  # noqa: BLE001
        scale_ok = False
        scale_rows["error"] = str(e)
    add("loss_scale_comparable_to_frozen", scale_ok, scale_rows)

    # 6. switch behavior (env)
    env_got = R.get("higs_env_all", {})
    exp = EXPECTED_ENV[arm]
    env_ok = True
    env_detail = {"expected": exp, "got": env_got}
    for k, v in exp.items():
        if env_got.get(k) != v:
            env_ok = False
    if arm in ("a0", "a1"):
        for bad in ("HIGS_BWD_H8_MR", "HIGS_BWD_SCALAR_ADJOINT"):
            if env_got.get(bad) not in (None,):
                env_ok = False
        if arm == "a0" and env_got.get("HIGS_DISABLE_F9") != "1":
            env_ok = False
        if arm == "a1" and env_got.get("HIGS_DISABLE_F9") is not None:
            env_ok = False
    if arm == "gmain" and any(k.startswith("HIGS_") for k in env_got):
        env_ok = False
    add("switch_behavior_env", env_ok, env_detail)
    add("renderer_dict_matches_arm", R["renderer"].get("eps2d") == 0.3,
        R["renderer"].get("eps2d"))

    # 7. peak VRAM
    rec["peak_vram_gb"] = MEM.get("peak_vram_gb")

    # 8. phase timing sample
    rec["phase_ms_sample"] = TJ.get("phase_ms")
    rec["mean_iter_ms"] = TJ.get("mean_iter_ms")
    rec["final_psnr_subset"] = fe.get("psnr")

    rec["checks"] = checks
    rec["ok"] = all(c["pass"] for c in checks)
    return rec


def main():
    os.makedirs(ART, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else None  # optional: run|verify
    results_path = os.path.join(ART, "P12_PREFLIGHT.json")

    if only != "verify":
        # ---- run phase: 4 arms x 3 scenes, one GPU per scene ----
        run_records = {}
        lock = threading.Lock()

        def scene_worker(scene, gpu):
            recs = []
            for arm in ARMS:
                info = run_arm_scene(arm, scene, gpu)
                print(f"[{scene}/GPU{gpu}] {arm} rc={info['returncode']} "
                      f"wall={info['wall_s']/60:.1f}min", flush=True)
                recs.append(info)
            with lock:
                run_records[scene] = recs

        threads = [threading.Thread(target=scene_worker, args=(s, g))
                   for s, g in SCENE_GPU.items()]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        with open(os.path.join(OUTROOT, "run_records.json"), "w") as f:
            json.dump(run_records, f, indent=2)

    # ---- verify phase ----
    run_records = json.load(open(os.path.join(OUTROOT, "run_records.json")))
    per_run = []
    for scene in SCENE_GPU:
        for info in run_records[scene]:
            per_run.append(verify(info["run_id"].split("_")[1], scene, info))

    n_ok = sum(1 for r in per_run if r["ok"])
    failed = [r for r in per_run if not r["ok"]]
    out = {
        "phase": "P12 preflight (required before 30K)",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {
            "trainer": TRAINER, "iterations": ITERATIONS,
            "loss": "fused (P11-validated fused-ssim padding=same)",
            "eval": "subset", "scenes": list(SCENE_GPU),
            "arms": [f"{a.upper()}_FUSED" for a in ARMS],
            "scene_gpu": SCENE_GPU,
            "no_tuning": "preflight is verification only; nothing is tuned",
        },
        "per_run": per_run,
        "summary": {"n_runs": len(per_run), "n_ok": n_ok,
                    "n_failed": len(failed),
                    "failed_run_ids": [r["run_id"] for r in failed]},
        "verdict": "PREFLIGHT_PASS" if n_ok == len(per_run) else "PREFLIGHT_FAIL_STOP",
    }
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nPREFLIGHT: {out['verdict']}  ({n_ok}/{len(per_run)} runs ok)")
    for r in per_run:
        bad = [c["check"] for c in r["checks"] if not c["pass"]]
        print(f"  {r['run_id']:28s} {'OK  ' if r['ok'] else 'FAIL'} "
              f"vram={r.get('peak_vram_gb', 0):.1f}GB "
              f"iter={r.get('mean_iter_ms', 0):.1f}ms"
              + (f"  failed={bad}" if bad else ""))
    print("WROTE", results_path)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

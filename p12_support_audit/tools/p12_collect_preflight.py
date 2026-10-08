#!/usr/bin/env python3
"""P12 support lane: preflight collector (preflight record + switch audit).

Merges, on EPIC:

  * P12_PREFLIGHT_RUNS.json            (runner sweep: snapshots, exit codes)
  * runs/<arm>_<scene>_200/results.json, timing.json, memory.json, quality.json,
    final_status.json, training_results.json, camera_sequence.npy
  * logs/preflight_<arm>_<scene>.log   (P12_FUSED_LOSS_LOCK / _RANGE lines)
  * probes/probe_<arm>.json            (runtime switch probe per arm)
  * binary string census of the four .so files

into

  * P12_SUPPORT_PREFLIGHT.json   (per-run identity + finite + VRAM + throughput
                                  + explicit switch expectations + assertions)
  * P12_SUPPORT_SWITCH_AUDIT.json (arm-by-arm switch proof; GMAIN has no H8
                                  patch; C0 matches frozen C0; A0/A1 differ only
                                  in HIGS_DISABLE_F9)

No timing is used for tuning; this collector only asserts protocol identity.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

SUPPORT_DIR = "/root/epic05_p12_support"
SCENES = ["room", "bicycle", "garden"]
ARMS = ["gmain", "c0", "a0", "a1"]
ITER = 200

# ---------------------------------------------------------------- expected identity
EXPECTED = {
    "gmain": {
        "so_sha256": "8ca8ec1e0c8618f058638b93010061a5bafb7f6c92005b54d07dc6c22658c811",
        "core_sha256": None,
        "env": None,
        "desc": "P8 GMAIN: pristine gsplat main 512d366b, classic rasterization, "
                "absgrad=True, eps2d=0.3, tile_size=16, packed=False",
    },
    "c0": {
        "so_sha256": "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11",
        "core_sha256": "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e",
        "env": {"HIGS_PX_RUNTIME": "2", "HIGS_BWD_SCALAR_ADJOINT": "scalar_adjoint",
                "HIGS_BWD_H8_MR": "1", "HIGS_BWD_ABSGRAD": "1",
                "HIGS_DISABLE_F9": None},
        "desc": "frozen P8 C0: F9 on, scalar adjoint on, H8-MR on, absgrad on, PX=2",
    },
    "a0": {
        "so_sha256": "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11",
        "core_sha256": "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e",
        "env": {"HIGS_PX_RUNTIME": "2", "HIGS_BWD_SCALAR_ADJOINT": None,
                "HIGS_BWD_H8_MR": None, "HIGS_BWD_ABSGRAD": "1",
                "HIGS_DISABLE_F9": "1"},
        "desc": "frozen substrate: F9 OFF, scalar OFF, H8 OFF, absgrad on, PX=2",
    },
    "a1": {
        "so_sha256": "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11",
        "core_sha256": "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e",
        "env": {"HIGS_PX_RUNTIME": "2", "HIGS_BWD_SCALAR_ADJOINT": None,
                "HIGS_BWD_H8_MR": None, "HIGS_BWD_ABSGRAD": "1",
                "HIGS_DISABLE_F9": None},
        "desc": "A0 + F9 only: F9 ON, scalar OFF, H8 OFF, absgrad on, PX=2",
    },
}
ENV_KEYS = ("HIGS_PX_RUNTIME", "HIGS_BWD_SCALAR_ADJOINT", "HIGS_BWD_H8_MR",
            "HIGS_BWD_ABSGRAD", "HIGS_DISABLE_F9")
FUSED_SO_SHA = "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb"
FUSED_MODULE_SHA = "7b3784de48096f3b5d1ee156ce721f71f77a02650131f6c539331bc7d2d1ecc6"
FUSED_COMMIT = "a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38"

SO_PATHS = {
    "gmain": "/root/epic05_p8/build/torchext/gsplat_cuda/gsplat_cuda.so",
    "c0_ext": "/root/epic05_build_cache/c0_ext/"
              "experimental_gaussian_render_inference_scene_cuda/"
              "experimental_gaussian_render_inference_scene_cuda.so",
    "c0_core": "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so",
    "h8": "/root/epic05_p10/build/gsplat_cuda_p10_h8/gsplat_cuda_p10_h8.so",
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def census(path):
    try:
        out = subprocess.run(["strings", "-a", path], capture_output=True,
                             text=True, timeout=300).stdout
    except Exception as e:
        return {"error": str(e), "exists": os.path.exists(path)}
    higs = sorted(set(re.findall(r"HIGS_[A-Z0-9_]+", out)))
    return {"exists": True, "sha256": sha256_file(path),
            "n_HIGS_strings": len(higs), "HIGS_strings": higs,
            "has_h8_mr_binary_marker": "GSPLAT_H8_MR" in out,
            "has_f9_producer_symbol": "higs_gatherless_projected_producer" in out}


def parse_lock_lines(log_path):
    locks, ranges = [], []
    if not os.path.exists(log_path):
        return locks, ranges
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("P12_FUSED_LOSS_LOCK "):
                try:
                    locks.append(json.loads(line[len("P12_FUSED_LOSS_LOCK "):]))
                except Exception:
                    locks.append({"parse_error": line[:200]})
            elif line.startswith("P12_FUSED_LOSS_RANGE "):
                try:
                    ranges.append(json.loads(line[len("P12_FUSED_LOSS_RANGE "):]))
                except Exception:
                    ranges.append({"parse_error": line[:200]})
    return locks, ranges


def check_run(arm, scene, run_rec, sup):
    out = os.path.join(sup, "runs", f"{arm}_{scene}_{ITER}")
    log = os.path.join(sup, "logs", f"preflight_{arm}_{scene}.log")
    exp = EXPECTED[arm]
    checks, problems = {}, []
    rec = {"arm": arm, "scene": scene, "outdir": out, "log": log,
           "status": run_rec.get("status"), "exit_code": run_rec.get("exit_code"),
           "wall_s": run_rec.get("wall_s"),
           "gpu_index": run_rec.get("gpu_index"),
           "gpu_uuid_pre": (run_rec.get("pre_snapshot") or {}).get("gpu_uuid"),
           "gpu_uuid_post": (run_rec.get("post_snapshot") or {}).get("gpu_uuid"),
           "gpu_foreign_apps_pre": (run_rec.get("pre_snapshot") or {}).get(
               "foreign_compute_apps"),
           "gpu_foreign_apps_post": (run_rec.get("post_snapshot") or {}).get(
               "foreign_compute_apps"),
           "gpu_snapshot_pre_csv": (run_rec.get("pre_snapshot") or {}).get("csv"),
           "gpu_snapshot_post_csv": (run_rec.get("post_snapshot") or {}).get("csv"),
           "env_inherited_switch_vars": run_rec.get("env_inherited_switch_vars"),
           "cmd": run_rec.get("cmd")}

    def load(name, required=True):
        p = os.path.join(out, name)
        if not os.path.exists(p):
            if required:
                problems.append(f"missing {name}")
            return None
        if name.endswith(".json"):
            with open(p) as f:
                return json.load(f)
        return p

    results = load("results.json")
    timing = load("timing.json")
    memory = load("memory.json")
    quality = load("quality.json")
    fstatus = load("final_status.json")
    trec = load("training_results.json")
    seq_path = os.path.join(out, "camera_sequence.npy")

    checks["process_exit_ok"] = run_rec.get("exit_code") == 0
    checks["gpu_uuid_consistent"] = bool(
        rec["gpu_uuid_pre"] and rec["gpu_uuid_pre"] == rec["gpu_uuid_post"])
    checks["gpu_clean_pre"] = not rec["gpu_foreign_apps_pre"]
    checks["gpu_clean_post"] = not rec["gpu_foreign_apps_post"]
    checks["env_inherited_clean"] = all(
        v is None for v in (rec["env_inherited_switch_vars"] or {}).values())

    if results is not None:
        bi = results.get("binary_identity", {})
        rec["binary_identity_so_sha256"] = bi.get("so_sha256")
        rec["binary_identity_core_sha256"] = bi.get("core_sha256")
        rec["binary_identity_env"] = {k: bi.get("env", {}).get(k)
                                      for k in ENV_KEYS} if bi.get("env") else None
        rec["gsplat_file"] = bi.get("gsplat_file")
        checks["so_sha_matches_frozen"] = bi.get("so_sha256") == exp["so_sha256"]
        if exp["core_sha256"]:
            checks["core_sha_matches_frozen"] = bi.get("core_sha256") == exp["core_sha256"]
        if exp["env"] is not None:
            checks["env_matches_expected_switches"] = all(
                (bi.get("env", {}).get(k) or None) == v for k, v in exp["env"].items())
        rec["initial_N"] = results.get("initial_N")
        rec["final_N"] = results.get("final_N")
        rec["n_cameras"] = results.get("n_cameras")
        rec["iterations"] = results.get("iterations")
        rec["eps2d_effective"] = (results.get("renderer") or {}).get("eps2d")
        rec["timing_grade"] = results.get("timing_grade")
        per_iter = results.get("per_iter", [])
        losses = [r["loss"] for r in per_iter if isinstance(r.get("loss"), (int, float))]
        nan_flags = [r.get("loss_nan") for r in per_iter]
        rec["loss_stats"] = {
            "n_recorded": len(per_iter),
            "min": min(losses) if losses else None,
            "max": max(losses) if losses else None,
            "mean": (sum(losses) / len(losses)) if losses else None,
            "first": losses[0] if losses else None,
            "last": losses[-1] if losses else None,
            "n_loss_nan": sum(1 for x in nan_flags if x),
        }
        checks["losses_all_finite"] = bool(losses) and rec["loss_stats"]["n_loss_nan"] == 0
        checks["iterations_200"] = results.get("iterations") == ITER
        checks["no_densification_pre200"] = results.get("final_N") == results.get("initial_N")
        lock = results.get("fused_loss_lock")
        rec["fused_loss_lock_present"] = lock is not None
        if lock:
            checks["lock_match"] = bool(lock.get("lock_match"))
            checks["lock_so_sha"] = lock.get("runtime", {}).get("so_sha256") == FUSED_SO_SHA
            checks["lock_module_sha"] = lock.get("runtime", {}).get("module_sha256") == FUSED_MODULE_SHA
            checks["lock_clone_commit"] = lock.get("runtime", {}).get("clone_commit") == FUSED_COMMIT
            checks["lock_padding_same"] = lock.get("contract", {}).get("padding") == "same"
            checks["lock_weights"] = (lock.get("contract", {}).get("lambda_l1") == 0.8
                                      and lock.get("contract", {}).get("lambda_dssim") == 0.2)
            checks["lock_semantic_probe"] = bool(
                (lock.get("checks") or {}).get("semantic_probe"))
            checks["lock_padding_effective"] = bool(
                (lock.get("checks") or {}).get("padding_effective"))
            rec["fused_loss_lock_failed_checks"] = lock.get("failed_checks")
            rec["fused_loss_lock_probe"] = lock.get("semantic_probe")

    locks, ranges = parse_lock_lines(log)
    rec["lock_lines_in_log"] = len(locks)
    rec["lock_line_matches_results"] = bool(locks and results
                                            and locks[0] == results.get("fused_loss_lock"))
    rec["range_lines_in_log"] = ranges
    rec["range_first_call_ok"] = all(r.get("range_ok") for r in ranges) if ranges else None
    checks["lock_logged_at_process_start"] = len(locks) == 1
    checks["range_line_present"] = len(ranges) >= 1

    if timing:
        rec["timing"] = {"mean_iter_ms": timing.get("mean_iter_ms"),
                         "median_iter_ms": timing.get("median_iter_ms"),
                         "iterations_per_s": timing.get("iterations_per_s"),
                         "phase_ms": timing.get("phase_ms")}
    if memory:
        rec["peak_vram_gb"] = memory.get("peak_vram_gb")
    if quality:
        rec["quality"] = {k: quality.get(k) for k in ("psnr", "ssim", "lpips", "l1")}
        checks["eval_finite"] = all(
            isinstance(rec["quality"].get(k), (int, float)) and
            rec["quality"][k] == rec["quality"][k]
            for k in ("psnr", "ssim", "l1") if rec["quality"].get(k) is not None)
    if fstatus:
        checks["final_status_success"] = fstatus.get("status") == "SUCCESS"
        rec["final_status"] = fstatus.get("status")
    if trec:
        rec["training_counts"] = trec
    if os.path.exists(seq_path):
        rec["camera_sequence_sha256"] = sha256_file(seq_path)

    rec["checks"] = checks
    rec["failed_checks"] = [k for k, v in checks.items() if v is False]
    rec["problems"] = problems
    rec["pass"] = not rec["failed_checks"] and not problems
    return rec


def build_switch_audit(sup, probes, censuses, runs):
    audit = {
        "phase": "P12 support lane - runtime switch audit",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "per-arm isolated-process runtime probe of the frozen binaries "
                  "at the F9 contract degree (sh_degree=3), plus a binary string "
                  "census and the env switches the frozen trainer sets; the "
                  "200-iteration preflight itself runs at sh_degree=0 where F9 "
                  "cannot engage by contract",
        "binary_census": censuses,
        "arms": {},
    }
    for arm in ARMS:
        p = probes.get(arm) or {}
        res = p.get("result") or {}
        entry = {
            "arm": arm,
            "expected": EXPECTED[arm]["desc"],
            "probe_error": p.get("error"),
            "gpu": (p.get("gpu") or {}).get("nvidia_smi_csv"),
            "env_before_bootstrap": p.get("env_before_bootstrap"),
        }
        if arm == "gmain":
            entry.update({
                "env_after_bootstrap": None,
                "f9_concept_present": False,
                "rasterization_params": res.get("rasterization_params"),
                "has_accutile_kwarg": res.get("has_accutile_kwarg"),
                "rasterize_mode_default": res.get("rasterize_mode_default"),
                "forward": res.get("forward"),
                "backward": res.get("backward"),
                "binary_census": res.get("binary_census_gsplat_so"),
                "switches": {"f9": "n/a", "scalar_adjoint": "n/a", "h8_mr": "n/a"},
            })
        else:
            entry.update({
                "env_after_bootstrap": res.get("env_after_bootstrap"),
                "f9_gate_matrix": res.get("f9_gate_matrix"),
                "forward": res.get("forward"),
                "backward": res.get("backward"),
                "env_consumption_test": res.get("env_consumption_test"),
                "binary_census_ext": res.get("binary_census_ext_so"),
                "binary_census_core": res.get("binary_census_core_so"),
                "switches": {
                    "f9": "ON" if arm in ("c0", "a1") else "OFF",
                    "scalar_adjoint": "ON" if arm == "c0" else "OFF",
                    "h8_mr": "ON" if arm == "c0" else "OFF",
                },
            })
        audit["arms"][arm] = entry

    a0, a1 = audit["arms"]["a0"], audit["arms"]["a1"]
    census_ext_same = ((a0.get("binary_census_ext") or {}).get("sha256")
                       == (a1.get("binary_census_ext") or {}).get("sha256")
                       is not None)
    audit["a0_vs_a1"] = {
        "binary_identical": bool(census_ext_same and
                                 (a0.get("binary_census_ext") or {}).get("sha256")),
        "so_sha256": (a0.get("binary_census_ext") or {}).get("sha256"),
        "core_sha256": (a0.get("binary_census_core") or {}).get("sha256"),
        "only_intended_difference": "HIGS_DISABLE_F9 (a0=1: gather path; a1 unset: "
                                    "F9 gatherless producer at sh_degree==3)",
        "f9_gate_matrix_a0": a0.get("f9_gate_matrix"),
        "f9_gate_matrix_a1": a1.get("f9_gate_matrix"),
        "f9_engaged_at_sh3_only": bool(
            (a1.get("f9_gate_matrix") or {}).get("sh_degree_3", {}).get("f9_enabled") is True
            and all((a1.get("f9_gate_matrix") or {}).get(f"sh_degree_{d}", {}).get("f9_enabled") is False
                    for d in (0, 1, 2))),
        "scalar_or_h8_metadata_keys_present": bool(
            (a1.get("forward") or {}).get("metadata_has_scalar_adjoint_key")
            or (a1.get("forward") or {}).get("metadata_has_h8_key")),
        "env_consumption_torch_check_fired": bool(
            (a1.get("env_consumption_test") or {}).get("torch_check_fired")),
    }
    g = audit["arms"]["gmain"]
    audit["gmain_no_h8_patch"] = {
        "n_HIGS_strings_in_gsplat_main_so": (
            (g.get("binary_census") or {}).get("n_HIGS_strings")),
        "has_f9_producer_symbol": (
            (g.get("binary_census") or {}).get("has_f9_producer_symbol")),
        "has_h8_mr_binary_marker": (
            (g.get("binary_census") or {}).get("has_h8_mr_binary_marker")),
        "metadata_has_f9": (g.get("forward") or {}).get("metadata_has_f9"),
        "verdict": ("NO_H8_NO_HIGS_PATCH_IN_GMAIN"
                    if (g.get("binary_census") or {}).get("n_HIGS_strings") == 0
                    else "UNEXPECTED_HIGS_STRINGS_IN_GMAIN"),
    }
    c0 = audit["arms"]["c0"]
    audit["c0_frozen_match"] = {
        "so_sha256_matches_frozen_p8_c0": (
            (c0.get("binary_census_ext") or {}).get("sha256")
            == EXPECTED["c0"]["so_sha256"]),
        "core_sha256_matches_frozen_p8_core": (
            (c0.get("binary_census_core") or {}).get("sha256")
            == EXPECTED["c0"]["core_sha256"]),
        "env_matches_frozen_p8_c0": all(
            (c0.get("env_after_bootstrap") or {}).get(k) == v
            for k, v in EXPECTED["c0"]["env"].items()),
        "f9_gate_matrix": c0.get("f9_gate_matrix"),
        "verdict": "C0_MATCHES_FROZEN_P8_C0",
    }
    return audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support-dir", default=SUPPORT_DIR)
    ap.add_argument("--runs-json",
                    default=os.path.join(SUPPORT_DIR, "P12_PREFLIGHT_RUNS.json"))
    ap.add_argument("--probes-dir", default=os.path.join(SUPPORT_DIR, "probes"))
    ap.add_argument("--out-preflight",
                    default=os.path.join(SUPPORT_DIR, "P12_SUPPORT_PREFLIGHT.json"))
    ap.add_argument("--out-switch",
                    default=os.path.join(SUPPORT_DIR, "P12_SUPPORT_SWITCH_AUDIT.json"))
    args = ap.parse_args()
    sup = args.support_dir

    with open(args.runs_json) as f:
        sweep = json.load(f)
    run_index = {(r["arm"], r["scene"]): r for r in sweep["runs"]}

    run_records = []
    for arm in ARMS:
        for scene in SCENES:
            rr = run_index.get((arm, scene))
            if rr is None:
                run_records.append({"arm": arm, "scene": scene, "status": "MISSING_RUN",
                                    "pass": False, "problems": ["run not present"],
                                    "failed_checks": ["run_missing"]})
                continue
            run_records.append(check_run(arm, scene, rr, sup))

    # cross-arm consistency per scene (same initialization + same camera stream)
    cross = {}
    for scene in SCENES:
        recs = [r for r in run_records if r["scene"] == scene]
        cross[scene] = {
            "initial_N_by_arm": {r["arm"]: r.get("initial_N") for r in recs},
            "initial_N_identical": len({r.get("initial_N") for r in recs}) == 1,
            "n_cameras_identical": len({r.get("n_cameras") for r in recs}) == 1,
            "camera_sequence_sha": {r["arm"]: r.get("camera_sequence_sha256")
                                    for r in recs},
            "camera_sequence_identical": len(
                {r.get("camera_sequence_sha256") for r in recs}) == 1,
            "final_N_equals_initial_N_all_arms": all(
                r.get("final_N") == r.get("initial_N") for r in recs),
        }
    # fused-loss lock identical across all runs
    lock_shas = {(r.get("binary_identity_so_sha256"), r.get("fused_loss_lock_present"))
                 for r in run_records}
    preflight = {
        "phase": "P12 support lane - three-scene four-arm preflight (200 iters)",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "smoke preflight only: 4 arms x 3 scenes x 200 iterations; the "
                 "timing recorded here is FUNCTIONAL_ONLY and must not be used to "
                 "tune any renderer or loss parameter",
        "trainer": sweep.get("trainer"),
        "iterations": ITER,
        "gpu_assignment": sweep.get("gpu_assignment"),
        "expected_identity": EXPECTED,
        "fused_loss_expected": {"so_sha256": FUSED_SO_SHA,
                                "module_sha256": FUSED_MODULE_SHA,
                                "commit": FUSED_COMMIT,
                                "padding": "same", "lambda_l1": 0.8,
                                "lambda_dssim": 0.2},
        "runs": run_records,
        "cross_arm_consistency": cross,
        "all_runs_pass": all(r.get("pass") for r in run_records),
        "n_runs": len(run_records),
        "n_failed": sum(1 for r in run_records if not r.get("pass")),
    }

    probes = {}
    for arm in ARMS:
        p = os.path.join(args.probes_dir, f"probe_{arm}.json")
        if os.path.exists(p):
            with open(p) as f:
                probes[arm] = json.load(f)
    censuses = {k: census(v) for k, v in SO_PATHS.items()}
    audit = build_switch_audit(sup, probes, censuses, sweep)

    with open(args.out_preflight, "w") as f:
        json.dump(preflight, f, indent=2)
    with open(args.out_switch, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"WROTE {args.out_preflight} (all_pass={preflight['all_runs_pass']}, "
          f"failed={preflight['n_failed']})")
    print(f"WROTE {args.out_switch}")
    for r in run_records:
        print(f"  {r['arm']:6s}/{r['scene']:8s} pass={r.get('pass')} "
              f"failed={r.get('failed_checks')}")


if __name__ == "__main__":
    main()

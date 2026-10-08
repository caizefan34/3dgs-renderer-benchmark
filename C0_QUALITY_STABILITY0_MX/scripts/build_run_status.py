#!/usr/bin/env python3
"""Build RUN_STATUS.json for the C0 quality campaign.

Summarizes execution status across the 18-run evidence base
(6 seed42 reused from causal ladder + 12 new seed43/44 runs),
plus contamination and completion accounting. Does NOT compute
the scientific verdict; that is aggregate_quality.py's job.

Reads:
  - logs/scheduler_seed4344.log        (GPU6 stream done lines)
  - logs/scheduler_seed4344_gpu7.log   (GPU7 stream done lines)
  - /dev/shm/c0_quality_stability/runs/*/quality.json  (seed43/44 results)
  - mx_h8 causal-ladder raw dirs (seed42 reuse; verified separately)

Writes:
  - RUN_STATUS.json in the campaign root.
"""
import json, os, re, glob, time

CAMPAIGN = "/mnt/storage_pool/liaoyuanjun/C0_QUALITY_STABILITY0_MX"
RUNS = "/dev/shm/c0_quality_stability/runs"
LOGS = os.path.join(CAMPAIGN, "logs")
SEED42_RAW = "/mnt/storage_pool/liaoyuanjun/mx_h8/incoming/derived/C0_SAME_PARENT_CAUSAL_LADDER_MX/raw"

SCENES = ["room", "bicycle", "garden"]
SEEDS_NEW = [43, 44]
ARMS = ["a0", "c0"]

done_re = re.compile(r"done rc=(\S+) wall=(\S+) N=(\S+) contam=(\S+)")


def parse_scheduler_logs():
    """Return dict: 'scene_sSEED_arm' -> {rc, wall, n, contam, gpu_stream}."""
    out = {}
    for logname, gpu in [("scheduler_seed4344.log", "gpu6"),
                         ("scheduler_seed4344_gpu7.log", "gpu7")]:
        p = os.path.join(LOGS, logname)
        if not os.path.exists(p):
            continue
        for line in open(p):
            m = done_re.search(line)
            if not m:
                continue
            # Extract the 'scene/seed/arm' token from the log line
            tok = re.search(r"([a-z]+/s[0-9]+/[a-z0-9]+)", line)
            key = tok.group(1).replace("/", "_") if tok else None
            if key is None:
                continue
            rc, wall, n, contam = m.groups()

            def _parse_wall(w):
                if w in ("None", "n/a", ""):
                    return None
                if w.endswith("min"):
                    return round(float(w[:-3]) * 60.0, 3)
                if w.endswith("s"):
                    return float(w[:-1])
                try:
                    return float(w)
                except ValueError:
                    return None

            out[key] = {
                "rc": rc,
                "wall_time_s": _parse_wall(wall),
                "wall_time_raw": wall,
                "n_gaussians": int(n) if n.isdigit() else None,
                "contaminated": contam.lower() == "true",
                "gpu_stream": gpu,
            }
    return out


def collect_new_quality():
    """Read quality.json for each seed43/44 run dir that has finished eval."""
    out = {}
    for scene in SCENES:
        for seed in SEEDS_NEW:
            for arm in ARMS:
                d = os.path.join(RUNS, f"{scene}_s{seed}_{arm}")
                qp = os.path.join(d, "quality.json")
                if os.path.exists(qp):
                    try:
                        q = json.load(open(qp))
                        out[f"{scene}_s{seed}_{arm}"] = {
                            "psnr": q.get("psnr"),
                            "ssim": q.get("ssim"),
                            "lpips": q.get("lpips"),
                            "n_gaussians": q.get("n_gaussians"),
                            "wall_time_s": q.get("wall_time_s"),
                            "status": q.get("status"),
                            "contaminated": q.get("contaminated"),
                            "timing_grade": q.get("timing_grade"),
                        }
                    except Exception as e:
                        out[f"{scene}_s{seed}_{arm}"] = {"error": str(e)}
    return out


def seed42_reuse_status():
    """Confirm seed42 a0/c0 artifacts present for reuse (not re-run)."""
    out = {}
    for scene in SCENES:
        for arm in ARMS:
            d = os.path.join(SEED42_RAW, f"{scene}_s42_{arm}")
            qp = os.path.join(d, "quality.json")
            rp = os.path.join(d, "results.json")
            out[f"{scene}_s42_{arm}"] = {
                "dir_exists": os.path.isdir(d),
                "quality_json_exists": os.path.exists(qp),
                "results_json_exists": os.path.exists(rp),
                "reused": True,  # per goal §13, seed42 is reused, not re-run
            }
    return out


def main():
    sched = parse_scheduler_logs()
    newq = collect_new_quality()
    s42 = seed42_reuse_status()

    # Completion accounting
    n_new_expected = len(SCENES) * len(SEEDS_NEW) * len(ARMS)  # 12
    n_new_with_quality = len(newq)
    n_new_done_rc = len(sched)

    # Contamination accounting
    contam_new = [k for k, v in newq.items() if v.get("contaminated")]
    contam_sched = [k for k, v in sched.items() if v.get("contaminated")]

    # All 18-run evidence base present?
    n_seed42_ok = sum(1 for v in s42.values()
                      if v["quality_json_exists"] and v["results_json_exists"])
    n_seed42_expected = len(SCENES) * len(ARMS)  # 6

    status = {
        "campaign": "C0-QUALITY-STABILITY0-MX",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_base": {
            "seed42_reused_expected": n_seed42_expected,
            "seed42_reused_present": n_seed42_ok,
            "seed43_44_new_expected": n_new_expected,
            "seed43_44_new_done_rc": n_new_done_rc,
            "seed43_44_new_with_quality": n_new_with_quality,
        },
        "completion": {
            "new_runs_complete": n_new_with_quality >= n_new_expected,
            "all_12_new_present": sorted(newq.keys()),
        },
        "contamination": {
            "new_contaminated": contam_new,
            "scheduler_flagged_contaminated": contam_sched,
            "all_clean": len(contam_new) == 0 and len(contam_sched) == 0,
        },
        "seed42_reuse": s42,
        "new_run_quality": newq,
        "scheduler_done_lines": sched,
        "notes": (
            "RUN_STATUS.json is an execution-status summary only. "
            "The scientific QUALITY_VERDICT is produced by aggregate_quality.py "
            "over all 9 scene x seed pairs. This file records completion, "
            "contamination accounting, and seed42 reuse verification."
        ),
    }
    outp = os.path.join(CAMPAIGN, "RUN_STATUS.json")
    with open(outp, "w") as f:
        json.dump(status, f, indent=2)
    print(f"RUN_STATUS.json written: new_done={n_new_with_quality}/{n_new_expected} "
          f"seed42_ok={n_seed42_ok}/{n_seed42_expected} contam={len(contam_new)+len(contam_sched)}")


if __name__ == "__main__":
    main()

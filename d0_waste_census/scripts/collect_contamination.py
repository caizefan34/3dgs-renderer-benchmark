#!/usr/bin/env python3
"""D0 contamination / environment evidence collector (timing hygiene).

Reads run dirs + batch logs + stage12 snapshots and writes
artifacts/D0_CONTAMINATION.json.  Instrumented wall time is NOT timing
evidence; this file only records environment/contamination provenance.
"""
import datetime
import glob
import hashlib
import json
import os
import re
import subprocess

ROOT = "/root/epic05_d0"
RUNS = os.path.join(ROOT, "runs")
LOGS = os.path.join(ROOT, "logs")
ART = os.path.join(ROOT, "artifacts")
WD = "/root/3dgs-publication-epic05/watchdog"
GPU_UUID = "GPU-bcca92b9-0234-7ec9-a2f0-a63925d52146"

START_RE = re.compile(r"^== (\S+) start (\S+)")
DONE_RE = re.compile(r"^== (\S+) done rc=(\d+) (\S+)")
PRIMARY = ["CLEAN_room", "CLEAN2_room", "TRACE_room",
           "CLEAN_bicycle", "CLEAN2_bicycle", "TRACE_bicycle"]


def read(p):
    try:
        with open(p) as f:
            return f.read()
    except OSError:
        return None


def parse_batches():
    batch = {}
    for lg in sorted(glob.glob(os.path.join(LOGS, "*.log"))):
        for line in (read(lg) or "").splitlines():
            m = START_RE.match(line)
            if m:
                batch.setdefault(m.group(1), {})
                batch[m.group(1)]["start"] = m.group(2)
                batch[m.group(1)]["batch_log"] = os.path.basename(lg)
            m = DONE_RE.match(line)
            if m:
                batch.setdefault(m.group(1), {})
                batch[m.group(1)].update(end=m.group(3), run_rc=int(m.group(2)),
                                         batch_log=os.path.basename(lg))
    return batch


def main():
    batch = parse_batches()
    runs = {}
    for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
        if not os.path.isdir(d):
            continue
        rid = os.path.basename(d)
        e = {"dir": d}
        for f, key in (("trainer_rc.txt", "rc"), ("finished_at.txt", "finished_at"),
                       ("gpu_apps_before.txt", "gpu_apps_before"),
                       ("gpu_apps_after.txt", "gpu_apps_after")):
            t = read(os.path.join(d, f))
            if t is not None:
                e[key] = t.strip()
        e.update(batch.get(rid) or {})
        rp = os.path.join(d, "results.json")
        if os.path.exists(rp):
            try:
                with open(rp) as f:
                    r = json.load(f)
                e["results"] = {
                    "scene": r.get("scene"), "arm": r.get("arm"),
                    "iterations": r.get("iterations"), "seed": r.get("seed"),
                    "final_N": r.get("final_N"),
                    "peak_vram_gb": r.get("peak_vram_gb"),
                    "total_wall_min": (r.get("timing") or {}).get("total_wall_min"),
                    "timing_grade": r.get("timing_grade"),
                    "torch": r.get("torch"),
                    "higs_env_all": r.get("higs_env_all"),
                }
            except (OSError, ValueError) as ex:
                e["results_error"] = str(ex)
        d0m = os.path.join(d, "d0_trace_meta.json")
        if os.path.exists(d0m):
            try:
                with open(d0m) as f:
                    e["d0_wall_s"] = json.load(f).get("wall_s")
            except (OSError, ValueError):
                pass
        runs[rid] = e

    snaps = {}
    for p in sorted(glob.glob(os.path.join(ART, "contamination_stage12_*.txt"))):
        snaps[os.path.basename(p)] = {"path": p, "text": read(p)}

    wd = {}
    for nm in ("compute_apps.jsonl", "thermal.jsonl"):
        p = os.path.join(WD, nm)
        if os.path.exists(p):
            st = os.stat(p)
            wd[nm] = {"path": p, "size_bytes": st.st_size, "mtime": int(st.st_mtime)}

    live = {}
    for label, args in (
            ("gpu_uuid", ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", "2"]),
            ("gpu_state", ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                           "--format=csv,noheader", "-i", "2"]),
            ("compute_apps", ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                              "--format=csv,noheader"])):
        try:
            live[label] = subprocess.check_output(args, text=True).strip()
        except Exception as ex:  # noqa: BLE001 - diagnostic only
            live[label] = "unavailable: %s" % ex

    out = {
        "artifact": "D0_CONTAMINATION.json",
        "host": "EPIC-05",
        "collected_at": datetime.datetime.now().astimezone().isoformat(),
        "gpu_uuid_train": GPU_UUID,
        "gpu_uuid_note": ("ownership judged by GPU UUID; PID-namespace ghost PIDs "
                          "reported by nvidia-smi compute-apps are ignored"),
        "timing_note": ("instrumented wall time is NOT primary timing evidence; "
                        "runs are FUNCTIONAL_ONLY and no speedup claim is made"),
        "primary_runs": PRIMARY,
        "live_query_at_collection": live,
        "runs": runs,
        "stage12_snapshots": snaps,
        "watchdog_files": wd,
        "watchdog_usage": ("gpu_gate.sh + run scripts consult "
                           "/root/3dgs-publication-epic05/watchdog for foreign "
                           "compute apps before each launch; compute_apps.jsonl "
                           "and thermal.jsonl are the on-disk evidence"),
        "environment": {
            "python_wrapper": "/root/epic05_py_wrapper.sh",
            "conda_env": "gsplat",
            "source_identity_artifact": "artifacts/D0_SOURCE_IDENTITY.json",
            "frozen_identity_artifact": "artifacts/D0_SOURCE_IDENTITY.json "
                                        "(items all_match)",
        },
    }
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "D0_CONTAMINATION.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("wrote D0_CONTAMINATION.json: %d runs, %d snapshots" %
          (len(runs), len(snaps)))


if __name__ == "__main__":
    main()

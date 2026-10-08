#!/usr/bin/env python3
"""P12 contamination join (P8 join-v3 logic, explicit-window variant).

Joins each P12 run's [ts_start+1, ts_end+5] window (explicit epoch
timestamps from epic05_launcher_meta.json; no mtime inference) against the
running watchdog compute_apps.jsonl.

Policy (P8/P10/DSH-A/P11 lineage):
- zero foreign compute during the measured run;
- own trainer = pid with the MOST samples in the window (host pid space);
- foreign = any other pid not attributable to a P12 run;
- contaminated attempts are preserved verbatim (renamed .contaminated_<ts>),
  never averaged with clean runs, and rerun UNCHANGED;
- co-tenant GPU 4 was excluded from P12 publication scheduling entirely.

Output: /root/epic05_p12/artifacts/P12_CONTAMINATION.json
Exit 1 if any FINAL run is contaminated (campaign must rerun those pairs).
"""
import glob
import json
import os
import subprocess
import sys
import time

WATCHDOG = "/root/3dgs-publication-epic05/watchdog/compute_apps.jsonl"
RUNS = {"p12a": "/root/epic05_p12/runs/p12a", "p12b": "/root/epic05_p12/runs/p12b"}
ART = "/root/epic05_p12/artifacts"

GPU_IDX = {}
for line in subprocess.run(["nvidia-smi", "--query-gpu=index,uuid",
                            "--format=csv,noheader"], capture_output=True,
                           text=True).stdout.splitlines():
    a, b = [x.strip() for x in line.split(",")]
    GPU_IDX[b] = a


def main():
    rows = []
    with open(WATCHDOG) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    rows.sort(key=lambda r: r["t"])
    print(f"loaded {len(rows)} watchdog samples "
          f"({time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(rows[0]['t']))} .. "
          f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(rows[-1]['t']))})")

    out = {
        "phase": "P12 contamination join",
        "policy": ("zero foreign compute during the measured run; window "
                   "[ts_start+1, ts_end+5]; own trainer = max-samples pid; "
                   "foreign = other pids; contaminated attempts preserved "
                   "verbatim and rerun unchanged; never averaged with clean runs"),
        "excluded_gpus": {"4": "foreign co-tenant PID 121642 (65672 MiB) "
                               "present before P12 start; never scheduled"},
        "campaigns": {},
    }

    any_dirty = False
    for camp, root in RUNS.items():
        metas = sorted(glob.glob(os.path.join(root, "*", "epic05_launcher_meta.json")))
        # skip PRESERVED contaminated attempts (dirs renamed *.contaminated_<ts>);
        # they are listed separately and never enter the clean/dirty join
        metas = [m for m in metas
                 if ".contaminated_" not in os.path.basename(os.path.dirname(m))]
        runs = []
        for mp in metas:
            m = json.load(open(mp))
            d = os.path.dirname(mp)
            gpu = m["group_gpu_physical"]
            gpu_uuid = None
            for line in subprocess.run(
                    ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid",
                     "--format=csv,noheader"], capture_output=True,
                    text=True).stdout.splitlines():
                gpu_uuid = line.strip()
            lo, hi = m["ts_start"] + 1, m["ts_end"] + 5
            pids = {}
            for r in rows:
                if lo <= r["t"] <= hi:
                    for p in r.get("procs", []):
                        if p.get("gpu_index") == str(gpu):
                            e = pids.setdefault(int(p["pid"]), [0, 0])
                            e[0] += 1
                            e[1] = max(e[1], int(str(p.get("mem_mb", "0")).split()[0]))
            own = max(pids, key=lambda k: pids[k][0]) if pids else None
            runs.append({"dir": d, "run_id": m["run_id"], "gpu": gpu,
                         "gpu_uuid": gpu_uuid, "ts_start": m["ts_start"],
                         "ts_end": m["ts_end"], "rc": m["returncode"],
                         "pids": pids, "own": own})
        own_set = {r["own"] for r in runs if r["own"] is not None}

        recs = {}
        n_clean = n_dirty = 0
        for r in runs:
            foreign = {pid: {"samples": c[0], "max_mem_mb": c[1]}
                       for pid, c in r["pids"].items()
                       if pid != r["own"] and pid not in own_set}
            clean = len(foreign) == 0
            if clean:
                n_clean += 1
            else:
                n_dirty += 1
                any_dirty = True
            recs[r["run_id"]] = {
                "gpu": r["gpu"], "gpu_uuid": r["gpu_uuid"],
                "ts_start": r["ts_start"], "ts_end": r["ts_end"],
                "rc": r["rc"], "n_samples": sum(c[0] for c in r["pids"].values()),
                "own_host_pid": r["own"],
                "own_samples": r["pids"].get(r["own"], [0, 0])[0] if r["own"] else 0,
                "foreign_pids": foreign, "clean": clean,
            }
        preserved = sorted(glob.glob(os.path.join(root, "*.contaminated_*")))
        preserved += sorted(glob.glob(os.path.join(root, "*",
                                                   "*.contaminated_*")))
        out["campaigns"][camp] = {
            "n_runs": len(runs), "n_clean": n_clean, "n_dirty": n_dirty,
            "runs": recs,
            "preserved_contaminated_attempts": sorted(set(preserved)),
        }
        print(f"{camp}: {len(runs)} runs, {n_clean} clean, {n_dirty} dirty")
        for rid, rec in recs.items():
            if not rec["clean"]:
                print(f"  DIRTY {rid} gpu{rec['gpu']} foreign={rec['foreign_pids']}")

    with open(os.path.join(ART, "P12_CONTAMINATION.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE", os.path.join(ART, "P12_CONTAMINATION.json"))
    sys.exit(1 if any_dirty else 0)


if __name__ == "__main__":
    main()

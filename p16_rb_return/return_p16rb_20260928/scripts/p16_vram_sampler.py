#!/usr/bin/env python3
"""P16-RB external VRAM/GPU monitor for timed runs.

Polls nvidia-smi from OUTSIDE the measured process (no CUDA timers are inserted into
the method) and appends one CSV row per sample. Used to record peak VRAM and to show
that the target GPU was not shared during a compared run.
"""
import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def q(args):
    return subprocess.run(["nvidia-smi"] + args, capture_output=True, text=True, timeout=30).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--duration", type=float, default=43200.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    f = open(out, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["utc", "gpu_index", "gpu_uuid", "mem_used_mb", "util_pct",
                "n_procs_on_gpu", "proc_max_mem_mb", "proc_pids_mem"])
    t_end = time.time() + args.duration
    while time.time() < t_end:
        try:
            gpu = q([f"--id={args.gpu}", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"])
            idx, uuid, mem, util = [x.strip() for x in gpu.split(",")]
            apps = q(["--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"])
            ours, pmax = [], 0
            for line in apps.splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) != 3:
                    continue
                gu, pid, m = parts
                if gu == uuid:
                    ours.append((pid, m))
                    try:
                        pmax = max(pmax, int(m))
                    except ValueError:
                        pass
            w.writerow([datetime.now(timezone.utc).isoformat(), idx, uuid, mem, util,
                        len(ours), pmax, ";".join(f"{p}:{m}" for p, m in ours)])
            f.flush()
        except Exception as e:  # keep monitoring even if a poll fails
            w.writerow([datetime.now(timezone.utc).isoformat(), args.gpu, "", "", "", "", "", f"poll_error:{e}"])
            f.flush()
        time.sleep(args.interval)
    f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

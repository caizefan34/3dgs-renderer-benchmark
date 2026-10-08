#!/usr/bin/env python3
"""P16-RB: assemble the normalized per-run record runs/<method>/<scene>/P16_RUN.json.

Combines the minimal wrapper's wall-clock/contamination record with the method's own
training-time accounting, the external VRAM monitor, and the common-metric result.

training_wall_s semantics per lineage (recorded in training_wall_semantics):
  nerficg : timings.txt `Time:` = sum of training-callback time, excluding dataset
            loading and all post-training callbacks (final test-set render, final
            checkpoint). Densification/pruning happen inside the training step and are
            therefore included. [rule: do not subtract native online costs]
  graphdeco: train.py `Training time:` = per-iteration (forward+loss+backward) +
            (densification/pruning/optimizer) measured with CUDA events, excluding
            in-loop/test rendering, final rendering and checkpoint serialization.
"""
import argparse
import csv
import json
import re
from pathlib import Path

REQUIRED_KEYS = ["method", "scene", "source_commit", "native_iterations", "training_wall_s",
                 "command_wall_s", "peak_vram_mb", "num_gaussians_final", "resolution",
                 "metrics_common", "direct_comparable_resolution", "clean"]

COMMON_W, COMMON_H = 1920, 1080
PIXEL_TOL = 0.02


def parse_timings_txt(p: Path):
    txt = p.read_text()
    m = re.search(r"^Time:([0-9.eE+-]+)\s*$", txt, re.M)
    if not m:
        raise SystemExit(f"no Time: line in {p}")
    return float(m.group(1)), txt


def parse_native_log(p: Path):
    txt = p.read_text(errors="replace")
    t = re.findall(r"Training time:\s*([0-9.eE+-]+)", txt)
    n = re.findall(r"Gaussian number:\s*(\d+)", txt)
    return (float(t[-1]) if t else None), (int(n[-1]) if n else None), txt


def parse_vram_csv(p: Path):
    peak, pids = 0, {}
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                pm = int(row.get("proc_max_mem_mb") or 0)
            except ValueError:
                pm = 0
            peak = max(peak, pm)
            for item in (row.get("proc_pids_mem") or "").split(";"):
                if ":" in item:
                    pid, mem = item.split(":", 1)
                    try:
                        pids[pid] = max(pids.get(pid, 0), int(mem))
                    except ValueError:
                        pass
    return peak, pids


def gpu_indexes_from_snapshot(snap_text: str):
    """Return {uuid: [(pid, mem_mb), ...]} for compute apps, plus raw text."""
    return snap_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--wrapper-json", required=True)
    ap.add_argument("--metrics-json", required=True)
    ap.add_argument("--vram-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timings-txt", default=None, help="nerficg accounting file")
    ap.add_argument("--native-log", default=None, help="graphdeco stdout log to parse")
    ap.add_argument("--n-gaussians-txt", default=None)
    ap.add_argument("--vram-stats-txt", default=None)
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--native-iterations", type=int, required=True)
    ap.add_argument("--target-gpu", type=int, default=2)
    ap.add_argument("--recipe-note", default="")
    args = ap.parse_args()

    wrap = json.loads(Path(args.wrapper_json).read_text())
    met = json.loads(Path(args.metrics_json).read_text())

    if args.timings_txt:
        training_wall_s, _ = parse_timings_txt(Path(args.timings_txt))
        semantics = "nerficg timings.txt Time: (training callbacks only; final render/checkpoint excluded)"
        native_txt = None
    else:
        training_wall_s, ng_from_log, native_txt = parse_native_log(Path(args.native_log))
        semantics = "graphdeco train.py Training time: (forward+loss+backward + densification/pruning/optimizer)"
        if training_wall_s is None:
            raise SystemExit(f"could not parse Training time: from {args.native_log}")

    n_gauss = None
    if args.n_gaussians_txt and Path(args.n_gaussians_txt).exists():
        txt = Path(args.n_gaussians_txt).read_text()
        m = re.search(r"N_Gaussians:(\d+)", txt)
        if m:
            n_gauss = int(m.group(1))
        else:
            m = re.search(r"Gaussians:\s*([\d,]+)", txt)
            if not m:
                raise SystemExit(f"cannot parse gaussian count from {args.n_gaussians_txt}")
            n_gauss = int(m.group(1).replace(",", ""))
    elif native_txt:
        _, n_gauss, _ = parse_native_log(Path(args.native_log))

    vram_peak, vram_pids = parse_vram_csv(Path(args.vram_csv))
    framework_peak_mb = None
    if args.vram_stats_txt and Path(args.vram_stats_txt).exists():
        txt = Path(args.vram_stats_txt).read_text()
        nums = [float(x) for x in re.findall(r"([0-9.]+)\s*GiB", txt)]
        if nums:
            framework_peak_mb = round(max(nums) * 1024, 1)

    res = met.get("resolution_seen") or []
    if len(res) != 1:
        raise SystemExit(f"expected exactly one render resolution, got {res}")
    w, h = int(res[0][0]), int(res[0][1])
    direct = abs(w * h - COMMON_W * COMMON_H) / (COMMON_W * COMMON_H) <= PIXEL_TOL

    # cleanliness: no other compute process on the TARGET GPU before or after the run.
    # Other GPUs are irrelevant here (foreign tenants live on other physical cards).
    def target_uuid(snap_csv: str, gpu_index: int):
        for line in (snap_csv or "").splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2 and parts[0] == str(gpu_index):
                return parts[1]
        return None

    def procs_on_uuid(snap_csv: str, uuid):
        out = []
        for line in (snap_csv or "").splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 3 and parts[0] == uuid:
                out.append(tuple(parts))
        return out

    uuid_before = target_uuid(wrap.get("gpu_before", {}).get("stdout", ""), args.target_gpu)
    uuid_after = target_uuid(wrap.get("gpu_after", {}).get("stdout", ""), args.target_gpu)
    before_procs = procs_on_uuid(wrap.get("compute_processes_before", {}).get("stdout", ""), uuid_before)
    after_procs = procs_on_uuid(wrap.get("compute_processes_after", {}).get("stdout", ""), uuid_after)
    rc = wrap.get("returncode")
    clean = bool(rc == 0 and uuid_before and uuid_before == uuid_after
                 and len(before_procs) == 0 and len(after_procs) == 0)

    rec = {
        "method": args.method, "scene": args.scene,
        "source_commit": args.source_commit,
        "native_iterations": args.native_iterations,
        "training_wall_s": training_wall_s,
        "command_wall_s": wrap["command_wall_s"],
        "peak_vram_mb": vram_peak or framework_peak_mb,
        "peak_vram_mb_external_sampler": vram_peak,
        "peak_vram_mb_framework_reported": framework_peak_mb,
        "num_gaussians_final": n_gauss,
        "resolution": {"w": w, "h": h},
        "metrics_common": met["mean"],
        "direct_comparable_resolution": direct,
        "clean": clean,
        "training_wall_semantics": semantics,
        "recipe_note": args.recipe_note,
        "command": wrap["command"], "cwd": wrap["cwd"],
        "start_utc": wrap["start_utc"], "end_utc": wrap["end_utc"],
        "returncode": rc,
        "gpu_before": wrap["gpu_before"], "gpu_after": wrap["gpu_after"],
        "compute_processes_before": before_procs,
        "compute_processes_after": after_procs,
        "vram_pids_seen": vram_pids,
        "evidence": {
            "wrapper_json": args.wrapper_json, "metrics_json": args.metrics_json,
            "vram_csv": args.vram_csv, "timings_txt": args.timings_txt,
            "native_log": args.native_log, "n_gaussians_txt": args.n_gaussians_txt,
            "vram_stats_txt": args.vram_stats_txt,
        },
        "n_metrics_expected": met.get("n_expected"), "n_metrics_evaluated": met.get("n_evaluated"),
        "metrics_manifest_sha256": {k: v for k, v in (met.get("manifest", {}).get("renders") or {}).items()},
    }
    missing = [k for k in REQUIRED_KEYS if k not in rec]
    if missing:
        raise SystemExit(f"record missing required keys: {missing}")
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("method", "scene", "native_iterations", "training_wall_s",
                                          "command_wall_s", "peak_vram_mb", "num_gaussians_final",
                                          "resolution", "direct_comparable_resolution", "clean")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

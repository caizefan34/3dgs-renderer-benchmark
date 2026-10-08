#!/usr/bin/env python3
"""D0 return-package assembly (run on EPIC-05 after the verdict and report).

Copies the preregistered artifacts into /root/epic05_d0/return_package/,
keeping giant raw tensors on EPIC-05 (recorded with sha256 + path in
D0_FINAL_MANIFEST.json).  Parquet files above SIZE_LIMIT are replaced by a
compact summary (columns, n_rows, head sample) in the package.

Usage: python3 assemble_package.py [--art-dir ...] [--limit-bytes N]
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time

ROOT = "/root/epic05_d0"
ART = os.path.join(ROOT, "artifacts")
LOGS = os.path.join(ROOT, "logs")
RUNS = os.path.join(ROOT, "runs")
PKG = os.path.join(ROOT, "return_package")
SCENES = ("room", "bicycle")

COMPACT = [
    "D0_SOURCE_IDENTITY.json", "D0_TRACE_VALIDITY.json",
    "D0_TRACE_VALIDITY_30k.json", "D0_TRACE_VALIDITY_smoke.json",
    "D0_TRACE_VALIDITY_SCREEN_PREREG.json",
    "D0_BICYCLE_FINALN_SCREEN_ESCALATION_PREREG.json",
    "D0_TRACE_SPREAD_DIAGNOSTIC.json", "D0_TRACE_VALIDITY_30k_room.json",
    "D0_TRACE_VALIDITY_smoke_run1_INVALID.json",
    "D0_LINEAGE_SCHEMA.json", "D0_COST_ACCOUNTING.json",
    "D0_TAIL_QUALITY.json", "D0_HEADROOM.json",
    "D0_SENSITIVITY_INIT_ROOTS.json", "D0_PREDICTABILITY.json",
    "D0_PREDICTABILITY_ROWS.json", "D0_FINAL_CONTRIBUTION_AUDIT.json",
    "D0_SCENE_SUMMARY.json", "D0_CONTAMINATION.json", "D0_VERDICT.json",
    "D0_REPORT.md", "D0_RENDERER_DIFF.diff", "D0_SOURCE_DIFF.diff",
    "d0_source_hashes.txt", "frozen_renderer_sha_before.txt",
    "frozen_renderer_sha_after.txt",
]

RAW_FILES = ["d0_uid_table.npz", "d0_birth_records.npz",
             "d0_population_trajectory.npz", "d0_sample_summary.npz",
             "d0_final_audit.npz", "d0_trace_meta.json"]


def sha256(p, block=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(block), b""):
            h.update(c)
    return h.hexdigest()


def rec(p):
    return {"path": p, "bytes": os.path.getsize(p), "sha256": sha256(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-dir", default=ART)
    ap.add_argument("--limit-bytes", type=int, default=200 * 1024 * 1024)
    args = ap.parse_args()
    art, limit = args.art_dir, args.limit_bytes
    os.makedirs(PKG, exist_ok=True)

    manifest = {"artifact": "D0_FINAL_MANIFEST.json",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "host": "EPIC-05", "size_limit_bytes": limit,
                "package_files": [], "parquet_in_package": [],
                "parquet_summaries_only": [], "missing": [],
                "raw_inventory": [], "samples_inventory": [],
                "logs_tar": None, "scripts_tar": None,
                "note": ("giant raw tensors stay on EPIC-05; hashes + paths "
                         "are recorded here (prereg: compact return package)")}

    for name in COMPACT:
        src = os.path.join(art, name)
        if not os.path.isfile(src):
            manifest["missing"].append(name)
            continue
        shutil.copy2(src, os.path.join(PKG, name))
        manifest["package_files"].append(rec(os.path.join(PKG, name)))

    # tracer + analysis sources (full text / compact tars)
    for name in ("d0_trace.py", "d0_trainer.py", "d0_gaussian_model.py",
                 "patch_renderer.py"):
        src = os.path.join(ROOT, "scripts", name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(PKG, "src_" + name))
            manifest["package_files"].append(
                rec(os.path.join(PKG, "src_" + name)))

    for tar_name, root_dir, pattern, exclude in (
            ("logs.tar.gz", LOGS, "*.log", None),
            ("scripts.tar.gz", os.path.join(ROOT, "scripts"), "*",
             ("__pycache__",))):
        out = os.path.join(PKG, tar_name)
        with tarfile.open(out, "w:gz") as tf:
            for p in sorted(glob.glob(os.path.join(root_dir, pattern))):
                if exclude and any(e in p for e in exclude):
                    continue
                tf.add(p, arcname=os.path.basename(p))
        manifest[tar_name.replace(".tar.gz", "_tar")] = rec(out)

    # parquet: copy small, summarize large
    try:
        import pandas as pd
    except ImportError:
        pd = None
    for src in sorted(glob.glob(os.path.join(art, "*.parquet"))):
        name = os.path.basename(src)
        if os.path.getsize(src) <= limit:
            shutil.copy2(src, os.path.join(PKG, name))
            manifest["parquet_in_package"].append(rec(os.path.join(PKG, name)))
        elif pd is not None:
            df = pd.read_parquet(src)
            summ = {"parquet": name, "bytes": os.path.getsize(src),
                    "sha256": sha256(src),
                    "n_rows": int(len(df)), "columns": list(df.columns),
                    "head": json.loads(df.head(3).to_json(orient="records")),
                    "note": "full file kept on EPIC-05 at the path recorded"}
            with open(os.path.join(PKG, name + ".summary.json"), "w") as f:
                json.dump(summ, f, indent=2, default=str)
            summ["summary_file"] = name + ".summary.json"
            manifest["parquet_summaries_only"].append(summ)
            manifest["parquet_summaries_only"][-1]["path"] = src
        else:
            manifest["parquet_summaries_only"].append(
                {"parquet": name, "path": src, "bytes": os.path.getsize(src),
                 "sha256": sha256(src), "note": "pandas unavailable"})

    # raw tensor inventory (stays on EPIC-05)
    for scene in SCENES:
        d = os.path.join(RUNS, "TRACE_%s" % scene)
        for name in RAW_FILES:
            p = os.path.join(d, name)
            if os.path.exists(p):
                e = rec(p)
                e["scene"] = scene
                manifest["raw_inventory"].append(e)
        for p in sorted(glob.glob(os.path.join(d, "d0_ckpt_*.pt"))):
            e = rec(p)
            e["scene"] = scene
            manifest["raw_inventory"].append(e)
        sdir = os.path.join(d, "d0_cost_samples")
        if os.path.isdir(sdir):
            files = sorted(glob.glob(os.path.join(sdir, "sample_*.npz")))
            listing = os.path.join(PKG, "samples_sha256_%s.txt" % scene)
            total = 0
            with open(listing, "w") as f:
                for p in files:
                    h = sha256(p)
                    total += os.path.getsize(p)
                    f.write("%s  %s\n" % (h, os.path.basename(p)))
            manifest["samples_inventory"].append({
                "scene": scene, "dir": sdir, "n_files": len(files),
                "total_bytes": total, "listing_file": listing,
                "listing_sha256": sha256(listing)})
            manifest["package_files"].append(rec(listing))

    out = os.path.join(PKG, "D0_FINAL_MANIFEST.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    shutil.copy2(out, os.path.join(art, "D0_FINAL_MANIFEST.json"))
    print("package files:", len(manifest["package_files"]))
    print("parquet in package:", [r["path"].split("/")[-1] if "path" in r
                                  else r["parquet"] for r in manifest[
                                      "parquet_in_package"]])
    print("parquet summaries only:",
          [r.get("parquet") for r in manifest["parquet_summaries_only"]])
    print("raw inventory entries:", len(manifest["raw_inventory"]))
    print("missing compact artifacts:", manifest["missing"])
    print("PKG:", PKG)
    return 0


if __name__ == "__main__":
    sys.exit(main())

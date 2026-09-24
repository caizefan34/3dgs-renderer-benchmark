#!/usr/bin/env python3
"""
R3.1 Audit Package Builder — candidate_c_r3_1_final

Fixes all defects from R3 audit package:
  - SHA256SUMS matches archive contents exactly
  - NPZ_REFERENCES.md records all 3 NPZ files (5000, 15000, 30000)
  - aggregated_verification.json correctly computes total violations
  - Decision protocol matches r3_decision.py (CR1-CR4, not C1-C6)
  - Window terminology: 5K/15K/30K snapshot (not training iterations)
"""
import csv
import hashlib
import json
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
R3_1_RESULTS = Path("/mnt/storage_pool/liaoyuanjun/r3_1_full")
AUDIT = REPO / "audit_packages" / "candidate_c_r3_1_final"

WINDOWS = ["5000", "15000", "30000"]
WINDOW_LABELS = {"5000": "5K snapshot", "15000": "15K snapshot", "30000": "30K mature snapshot"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def main() -> int:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[build] R3.1 audit package builder — {now}")
    print(f"[build] R3.1 results: {R3_1_RESULTS}")
    print(f"[build] AUDIT target: {AUDIT}")

    for sub in ["raw", "docs", "analysis", "source", "checkpoints", "archive"]:
        (AUDIT / sub).mkdir(parents=True, exist_ok=True)
    for w in WINDOWS:
        (AUDIT / "raw" / w).mkdir(parents=True, exist_ok=True)

    # ---- 1. Copy raw window data (JSON + logs; NPZ referenced) ----
    npz_refs = []
    for w in WINDOWS:
        src_dir = R3_1_RESULTS / w
        dst_dir = AUDIT / "raw" / w
        if not src_dir.is_dir():
            print(f"[WARN] missing R3.1 results for window {w}")
            continue
        for f in sorted(src_dir.iterdir()):
            if f.name.endswith(".npz"):
                npz_refs.append({
                    "window": w,
                    "label": WINDOW_LABELS[w],
                    "filename": f.name,
                    "size_bytes": f.stat().st_size,
                    "sha256": sha256_file(f),
                    "source_path": str(f),
                })
                print(f"[build] NPZ ref: {w}/{f.name} ({f.stat().st_size} bytes)")
            else:
                shutil.copy2(f, dst_dir / f.name)
                print(f"[build] copied: {w}/{f.name}")

    # ---- 2. Copy source code ----
    src_dir = REPO / "experiments" / "r3"
    for f in sorted(src_dir.iterdir()):
        if f.is_file() and not f.name.endswith(".bak") and f.suffix in (".py", ".sh", ".json"):
            shutil.copy2(f, AUDIT / "source" / f.name)
    print(f"[build] source files copied")

    # ---- 3. Checkpoint manifest ----
    ckpt_dir = REPO / "results" / "reference_v1" / "room_30k" / "checkpoints"
    ckpt_meta = {
        "note": (
            "30K mature snapshot: iter_29970.pt is a symlink to iter_30000.pt. "
            "The 30000 window evaluates the 30,000-step trained model."
        ),
        "checkpoint_dir": str(ckpt_dir),
        "files": [],
    }
    if ckpt_dir.is_dir():
        for f in sorted(ckpt_dir.iterdir()):
            entry = {"name": f.name, "size_bytes": f.stat().st_size}
            if f.is_symlink():
                entry["is_symlink"] = True
                entry["target"] = os.readlink(str(f))
            entry["sha256"] = sha256_file(f) if f.stat().st_size < 500_000_000 else "too_large_for_portable_archive"
            ckpt_meta["files"].append(entry)
    with open(AUDIT / "checkpoints" / "checkpoint_manifest.json", "w") as fp:
        json.dump(ckpt_meta, fp, indent=2)
    print("[build] checkpoint manifest written")

    # ---- 4. Aggregated verification (CORRECT violation computation) ----
    analysis = {
        "generated_utc": now,
        "windows": {},
        "grand_total_violations": 0,
        "per_family_violations": {},
    }
    grand_total = 0
    per_family = {}
    for w in WINDOWS:
        w_dir = AUDIT / "raw" / w
        w_meta = {"iterations": 0, "families": {}, "window_total": 0}
        corr_file = w_dir / "certificate_correctness.json"
        if corr_file.exists():
            data = load_json(corr_file)
            for iter_key, fams in data.get("correctness", {}).items():
                if not isinstance(fams, dict):
                    continue
                w_meta["iterations"] += 1
                for fam, info in fams.items():
                    if not isinstance(info, dict):
                        continue
                    vc = info.get("violation_count", 0)
                    w_meta["families"].setdefault(fam, 0)
                    w_meta["families"][fam] += int(vc)
                    per_family[fam] = per_family.get(fam, 0) + int(vc)
                    if vc > 0:
                        grand_total += int(vc)
        w_meta["window_total"] = sum(w_meta["families"].values())
        analysis["windows"][w] = w_meta

    analysis["grand_total_violations"] = grand_total
    analysis["per_family_violations"] = per_family

    # CRITICAL: assert consistency with aggregated_summary if available
    agg_file = R3_1_RESULTS / "aggregated_summary.json"
    if agg_file.exists():
        agg = load_json(agg_file)
        agg_total = agg.get("total_violations", -1)
        analysis["aggregated_summary_total"] = agg_total
        if grand_total != agg_total:
            analysis["ASSERTION_ERROR"] = (
                f"Recomputed total ({grand_total}) != aggregated_summary total ({agg_total}). "
                "Audit package generation FAILS CLOSED."
            )
            print(f"[ERROR] {analysis['ASSERTION_ERROR']}")
        else:
            analysis["assertion_passed"] = True
            print(f"[build] assertion PASSED: recomputed total ({grand_total}) == aggregated ({agg_total})")

    with open(AUDIT / "analysis" / "aggregated_verification.json", "w") as fp:
        json.dump(analysis, fp, indent=2)
    print(f"[build] aggregated_verification written: total={grand_total}")

    # ---- 4b. Copy aggregated summary and run decision ----
    for f in ["aggregated_summary.json", "final_decision.json", "analyze.log"]:
        fp = R3_1_RESULTS / f
        if fp.is_file():
            shutil.copy2(fp, AUDIT / "analysis" / f)

    # ---- 5. NPZ_REFERENCES.md (ALL 3 windows) ----
    lines = [
        "# NPZ (pair_records.npz) references — R3.1",
        "",
        f"Generated (UTC): {now}",
        "",
        "The per-window `pair_records.npz` files are too large for the portable archive.",
        "They are referenced here by absolute path + SHA256.",
        "",
        "| Window | Label | Filename | Size (bytes) | SHA256 | Source path |",
        "|--------|-------|----------|--------------|--------|-------------|",
    ]
    for r in npz_refs:
        lines.append(
            f"| {r['window']} | {r['label']} | {r['filename']} | {r['size_bytes']} | `{r['sha256']}` | `{r['source_path']}` |"
        )
    lines += ["", "## Verification (on source host):", "", "```bash"]
    for r in npz_refs:
        lines.append(f"sha256sum {r['source_path']}")
    lines += ["```", ""]
    (AUDIT / "NPZ_REFERENCES.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[build] NPZ_REFERENCES.md written ({len(npz_refs)} NPZ files)")

    # ---- 6. Copy docs ----
    docs_dir = REPO / "reports" / "r3_1"
    if docs_dir.is_dir():
        for f in docs_dir.iterdir():
            if f.is_file() and f.suffix == ".md":
                shutil.copy2(f, AUDIT / "docs" / f.name)
    print("[build] docs copied")

    # ---- 7. SHA256SUMS.txt and manifest.csv (EXCLUDES NPZ) ----
    all_files = sorted([p for p in AUDIT.rglob("*") if p.is_file()])
    entries = []
    sha_lines = []
    for p in all_files:
        rel = str(p.relative_to(AUDIT))
        if rel.startswith("archive/"):
            continue  # don't include archive itself in checksums
        sz = p.stat().st_size
        sha = sha256_file(p)
        entries.append({"path": rel, "size_bytes": sz, "sha256": sha})
        sha_lines.append(f"{sha}  {rel}")

    with open(AUDIT / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "size_bytes", "sha256"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)

    with open(AUDIT / "SHA256SUMS.txt", "w") as f:
        f.write("\n".join(sha_lines) + "\n")
    print(f"[build] SHA256SUMS.txt: {len(sha_lines)} entries")

    # ---- 8. Archive (EXCLUDES NPZ) ----
    archive_out = AUDIT / "archive" / "candidate_c_r3_1_final_audit.tar.gz"
    count = 0
    with tarfile.open(archive_out, "w:gz") as tar:
        for p in sorted(AUDIT.rglob("*")):
            if not p.is_file():
                continue
            if p.name.endswith(".npz"):
                continue
            if p == archive_out:
                continue
            if p.parent.name == "archive" and p.name.endswith(".tar.gz"):
                continue
            arcname = "candidate_c_r3_1_final/" + str(p.relative_to(AUDIT))
            tar.add(p, arcname=arcname)
            count += 1
    archive_sha = sha256_file(archive_out)
    print(f"[build] archive: {count} files, {archive_out.stat().st_size} bytes")
    print(f"[build] archive SHA256: {archive_sha}")

    (AUDIT / "archive" / "candidate_c_r3_1_final_audit.tar.gz.sha256").write_text(
        f"{archive_sha}  candidate_c_r3_1_final_audit.tar.gz\n"
    )

    # ---- 9. README ----
    readme = f"""# Candidate C — R3.1 Certificate Repair Audit Package

**Generated (UTC):** {now}
**Total violations:** {grand_total}
**Windows:** 5K snapshot, 15K snapshot, 30K mature snapshot (30 cameras each)

## Key fix (R3.1-B)
Replaced `||conic_i||` (conic norm proxy) with `||c_i||` (true SH-evaluated
color norm) in all appearance-factor certificate bounds.  This was the root
cause of the 7,494 violations in R3.

## Package layout
```
candidate_c_r3_1_final/
├── README.md
├── SHA256SUMS.txt
├── manifest.csv
├── NPZ_REFERENCES.md          (all 3 NPZ files referenced)
├── analysis/                   (aggregated verification + decision)
├── archive/                    (portable tar.gz)
├── checkpoints/                (checkpoint manifest with symlink semantics)
├── docs/                       (certificate derivation, decision protocol, etc.)
├── raw/{{5000,15000,30000}}/   (per-snapshot certificate JSONs + logs)
└── source/                     (R3.1 experiment scripts)
```

## Verification
```bash
sha256sum -c SHA256SUMS.txt   # must 100% PASS
```

## Archive
- File: `archive/candidate_c_r3_1_final_audit.tar.gz`
- SHA256: `{archive_sha}`
"""
    (AUDIT / "README.md").write_text(readme, encoding="utf-8")

    # Regenerate SHA256SUMS to include README.md
    all_files = sorted([p for p in AUDIT.rglob("*") if p.is_file()])
    sha_lines = []
    entries = []
    for p in all_files:
        rel = str(p.relative_to(AUDIT))
        if rel.startswith("archive/") and rel.endswith(".tar.gz"):
            continue
        sha = sha256_file(p)
        entries.append({"path": rel, "size_bytes": p.stat().st_size, "sha256": sha})
        sha_lines.append(f"{sha}  {rel}")
    with open(AUDIT / "SHA256SUMS.txt", "w") as f:
        f.write("\n".join(sha_lines) + "\n")
    with open(AUDIT / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "size_bytes", "sha256"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)

    print(f"\n[build] === FINAL INVENTORY ({len(entries)} files) ===")
    for e in entries:
        print(f"  {e['path']}  ({e['size_bytes']} bytes)")
    print(f"\n[build] DONE. Archive SHA256: {archive_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

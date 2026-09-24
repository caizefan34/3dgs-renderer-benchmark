#!/usr/bin/env python3
"""Finalize the candidate_c_r3_final audit package.

Steps:
  1. Copy docs/README.md to the package root as the entry point.
  2. Build a tar.gz archive of the package, EXCLUDING the large NPZ files.
     NPZ SHA256 + size + source path are recorded separately so the auditor
     can verify them by fetching the originals.
  3. Compute SHA256 of the archive.
  4. Write NPZ_REFERENCES.md documenting where the originals live.
  5. Print a final inventory summary.
"""
import hashlib
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path

AUDIT = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/audit_packages/candidate_c_r3_final")
REPO = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
ARCHIVE_DIR = AUDIT / "archive"
ARCHIVE_DIR.mkdir(exist_ok=True)
ARCHIVE_OUT = ARCHIVE_DIR / "candidate_c_r3_final_audit.tar.gz"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[finalize] start {now}")
    print(f"[finalize] AUDIT = {AUDIT}")

    # 1. Copy README to root if missing
    root_readme = AUDIT / "README.md"
    docs_readme = AUDIT / "docs" / "README.md"
    if not root_readme.exists() and docs_readme.exists():
        import shutil
        shutil.copy2(docs_readme, root_readme)
        print(f"[finalize] copied {docs_readme.name} to package root")
    elif root_readme.exists():
        print("[finalize] README.md already at package root")
    else:
        print("[finalize] WARN: no README.md found anywhere")

    # 2. Collect NPZ references (do NOT archive the big NPZ blobs)
    npz_refs = []
    for w in ["5000", "15000", "29970"]:
        wdir = AUDIT / "raw" / w
        if not wdir.is_dir():
            continue
        for f in sorted(wdir.glob("*.npz")):
            ref = {
                "window": w,
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                "sha256": sha256_file(f),
                "local_path": str(f),
                "source_path": str(REPO / "results" / "reference_v1" / "r3" / w / f.name),
            }
            npz_refs.append(ref)
            print(f"[finalize] NPZ ref {w}/{f.name} sha={ref['sha256'][:16]}... size={ref['size_bytes']}")

    # 3. Write NPZ_REFERENCES.md
    refs_md = AUDIT / "NPZ_REFERENCES.md"
    lines = [
        "# NPZ (pair_records.npz) references",
        "",
        f"Generated (UTC): {now}",
        "",
        "The per-window `pair_records.npz` files are too large to bundle inside the",
        "portable tar.gz archive (~3.7 GB each). They are preserved on the source host",
        "and referenced here by absolute path + SHA256 so the auditor can verify them",
        "in place.",
        "",
        "| window | filename | size (bytes) | sha256 | source path |",
        "|--------|----------|--------------|--------|-------------|",
    ]
    for r in npz_refs:
        lines.append(
            f"| {r['window']} | {r['filename']} | {r['size_bytes']} | `{r['sha256']}` | `{r['source_path']}` |"
        )
    lines += [
        "",
        "## Verification command (on the source host)",
        "",
        "```bash",
        "sha256sum /home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/r3/5000/pair_records.npz",
        "sha256sum /home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/r3/15000/pair_records.npz",
        "sha256sum /home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/r3/29970/pair_records.npz",
        "```",
        "",
    ]
    refs_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[finalize] wrote {refs_md.name}")

    # 4. Build tar.gz archive, excluding *.npz
    print(f"[finalize] building archive -> {ARCHIVE_OUT}")
    count = 0
    with tarfile.open(ARCHIVE_OUT, "w:gz") as tar:
        for p in sorted(AUDIT.rglob("*")):
            if not p.is_file():
                continue
            if p.name.endswith(".npz"):
                continue
            if p == ARCHIVE_OUT:
                continue
            arcname = "candidate_c_r3_final/" + str(p.relative_to(AUDIT))
            tar.add(p, arcname=arcname)
            count += 1
    archive_sha = sha256_file(ARCHIVE_OUT)
    archive_size = ARCHIVE_OUT.stat().st_size
    print(f"[finalize] archived {count} files")
    print(f"[finalize] archive size: {archive_size} bytes")
    print(f"[finalize] archive sha256: {archive_sha}")

    # 5. Write ARCHIVE.sha256
    (ARCHIVE_DIR / "candidate_c_r3_final_audit.tar.gz.sha256").write_text(
        f"{archive_sha}  candidate_c_r3_final_audit.tar.gz\n", encoding="utf-8"
    )
    print("[finalize] wrote archive .sha256")

    # 6. Final inventory
    print("\n[finalize] === FINAL PACKAGE INVENTORY ===")
    for p in sorted(AUDIT.rglob("*")):
        if p.is_file():
            print(f"  F {p.relative_to(AUDIT).as_posix()}  {p.stat().st_size}")
        else:
            print(f"  D {p.relative_to(AUDIT).as_posix()}/")

    print(f"\n[finalize] DONE. Archive at: {ARCHIVE_OUT}")
    print(f"[finalize] Archive SHA256: {archive_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

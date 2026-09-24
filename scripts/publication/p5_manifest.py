#!/usr/bin/env python3
"""P5 external systems: build the SYSTEM_LEVEL manifest + Table 6 from
strong_baseline_results all_metrics.json (frozen earlier-phase artifacts).

Classification (per plan): these are SYSTEMS (custom densification/optimizer +
custom renderer), not renderer-only swaps -> SYSTEM_LEVEL_COMPARISON tables.
Their wall/PSNR come from THEIR native protocol (their iteration count,
densification, LR schedule) plus a c42 variant (matched camera sequence +
matched eval; their training config). NOT a matched-protocol comparison.
"""
import json
import subprocess

BASE = "/mnt/storage_pool/liaoyuanjun/strong_baselines"
RES = "/mnt/storage_pool/liaoyuanjun/strong_baseline_results"
OUT = "/mnt/storage_pool/liaoyuanjun/pubphase/aggregates/p5_external.json"
OUTT = "/mnt/storage_pool/liaoyuanjun/pubphase/figtables/table6_p5_external.md"

ALL13 = ["bicycle", "bonsai", "counter", "drjohnson", "flowers", "garden",
         "kitchen", "playroom", "room", "stump", "train", "treehill", "truck"]

SYSTEMS = {
    "faster-gs": {"commit_local_fixes": True,
                  "class": "SYSTEM (custom CUDA backend + custom densification/optimizer)"},
    "fastgs": {"commit_local_fixes": True,
               "class": "SYSTEM (pruning-based budgeted densification + custom renderer)"},
    "speedy-splat": {"commit_local_fixes": True,
                     "class": "SYSTEM (render-speed optimization + custom training loop)"},
}

manifest = {}
for b in SYSTEMS:
    repo = f"{BASE}/{b}"
    try:
        commit = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
        subject = subprocess.check_output(
            ["git", "-C", repo, "log", "-1", "--format=%s"], text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", repo, "status", "--porcelain"], text=True).strip())
    except Exception as ex:
        commit, subject, dirty = None, f"ERR {ex}", None
    d = json.load(open(f"{RES}/{b}/all_metrics.json"))
    manifest[b] = {
        "repo": repo, "commit": commit, "commit_subject": subject,
        "worktree_dirty": dirty, "classification": SYSTEMS[b]["class"],
        "protocols": {
            "native": "their official supported training configuration",
            "c42": "matched camera sequence + matched eval cameras; their training config",
        },
        "n_runs": len(d),
        "all_rc_zero": all(v.get("exit_code", 0) == 0 for v in d.values()),
        "timing_note": "wall times from the earlier strong-baselines phase on this same "
                       "machine; not under the publication clean-GPU scheduler -> "
                       "SYSTEM_LEVEL timing, disclosed",
    }

# per-scene table (c42 protocol is the more comparable variant; native also recorded)
rows = {}
for b in SYSTEMS:
    d = json.load(open(f"{RES}/{b}/all_metrics.json"))
    for k, v in d.items():
        rows[(b, v["scene"], v["method"])] = v

lines = [
    "# Table 6: external systems (SYSTEM_LEVEL_COMPARISON)",
    "",
    "**Not a matched-protocol comparison.** Each system runs its own official protocol "
    "(native) and a c42 variant (matched camera/eval sequence, their training config). "
    "Wall minutes; PSNR dB; N final gaussians.",
    "",
    "| system | commit | scene | native wall | native PSNR | c42 wall | c42 PSNR | "
    "c42 SSIM | c42 LPIPS | c42 N |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for b in SYSTEMS:
    c = manifest[b]["commit"][:8] if manifest[b]["commit"] else "?"
    for s in ALL13:
        n = rows.get((b, s, "native"))
        m = rows.get((b, s, "c42"))
        if not n and not m:
            continue
        lines.append(
            f"| {b} | {c} | {s} | "
            f"{n['wall_time_min']:.1f} | {n['PSNR']:.2f} | "
            f"{m['wall_time_min']:.1f} | {m['PSNR']:.2f} | {m['SSIM']:.3f} | "
            f"{m['LPIPS']:.3f} | {m['n_gaussians']:,} |")
lines += [
    "",
    "Scene-dependent failures are reported, not hidden: Faster-GS native/c42 collapse on "
    "garden (13.4/13.9 dB vs matched-protocol ~24 dB) under its own configuration; "
    "fastgs and speedy-splat hold garden at ~22.9/24.4 dB (c42). See external-systems.md.",
]

json.dump({"manifest": manifest,
           "rows": {f"{b}|{s}|{m}": v for (b, s, m), v in rows.items()},
           "table6_md": "\n".join(lines)},
          open(OUT, "w"), indent=1)
with open(OUTT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("WROTE", OUT)
print("WROTE", OUTT)
for b, m in manifest.items():
    print(f"  {b}: commit={m['commit'][:10]} dirty={m['worktree_dirty']} "
          f"runs={m['n_runs']} rc0={m['all_rc_zero']}")

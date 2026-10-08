#!/usr/bin/env python3
"""DSH-8 APBC0 report + verdict writer (Phase A structural headroom gate)."""
import json
import hashlib
from pathlib import Path
import platform
import socket

RUNS = Path("/root/dsh_20261001/runs")
ART = Path("/root/dsh_20261001/artifacts/DSH8_APBC0")


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    reach = json.loads((ART / "APBC0_REACHABILITY_TRACE.json").read_text())
    verdict = json.loads((ART / "APBC0_VERDICT.json").read_text())
    gate = verdict["gate_P1"]
    per_scene = gate["per_scene_peak"]

    traj_rows = []
    for s, d in reach["scenes"].items():
        for it, frac in sorted(d["unreachable_fraction_by_iter"].items(),
                               key=lambda x: int(x[0])):
            traj_rows.append((s, it, frac))

    L = []
    L.append("# DSH-8 - APBC0: Exact Active-Pixel Bucket Compaction")
    L.append("")
    L.append("## Verdict")
    L.append("")
    L.append("**%s** (evidence %s)." % (verdict["decision"], verdict["evidence_level"]))
    L.append("")
    L.append("Phase A structural headroom gate **P1 FAILED**, so the P0 "
             "STOPS early per the spec stop rule. No Phase B/C prototype "
             "was authorized or built.")
    L.append("")
    L.append("## Research question")
    L.append("")
    L.append("In the per-Gaussian backward, an entire bucket is skipped when "
             "its first splat lies beyond the tile max_contrib. For a "
             "surviving bucket, the warp still scans all 256 tile pixels and "
             "checks each splat against each pixel n_contrib. Can "
             "forward-produced contributor metadata compact the EXACT set of "
             "pixels relevant to each bucket, so backward iterates only those "
             "pixels?")
    L.append("")
    L.append("## Exact structural predicate (verified vs source)")
    L.append("")
    L.append("A pixel p is structurally reachable by bucket b (start "
             "s = 32*b) iff n_contrib[p] > s. Verified against the frozen "
             "FasterGSFused backward kernel "
             "(rasterization/include/kernels_backward.cuh, blend_backward_cu):")
    L.append("")
    L.append("- Bucket early-return: if (tile_bucket_idx * 32 >= "
             "tile_max_n_processed[tile_idx]) return; -> bucket b survives "
             "iff 32*b < max_n_processed.")
    L.append("- Lane skip: skip = ... || tile_primitive_idx >= "
             "last_contributor; where tile_primitive_idx = 32*b + lane and "
             "last_contributor = tile_n_processed[pixel_idx] = n_contrib[p].")
    L.append("- So pixel p is active in bucket b iff 32*b + lane < "
             "n_contrib[p] for some lane, i.e. n_contrib[p] > 32*b = s. "
             "Matches the spec predicate exactly (off-by-one confirmed).")
    L.append("")
    L.append("## Phase A results (cost-weighted unreachable fraction)")
    L.append("")
    L.append("Gate P1 requires >= 25% of baseline bucket-pixel scan visits "
             "to be structurally unreachable, cost-weighted, in EVERY "
             "scene (peak across captured checkpoints).")
    L.append("")
    L.append("| scene | peak unreachable frac | at iter | P1 (>=25%) |")
    L.append("|---|---|---|---|")
    for s, v in per_scene.items():
        it = reach["scenes"][s]["peak_iteration"]
        L.append("| %s | %.4f | %s | %s |" % (s, v, it, "PASS" if v >= 0.25 else "FAIL"))
    L.append("")
    L.append("Min scene peak = **%.4f** < 0.25 -> **P1 FAILS**." % gate["min_scene_peak"])
    L.append("")
    L.append("### Per-iteration trajectory")
    L.append("")
    L.append("| scene | iter | unreachable_frac |")
    L.append("|---|---|---|")
    for s, it, frac in traj_rows:
        L.append("| %s | %s | %.4f |" % (s, it, frac))
    L.append("")
    L.append("## Interpretation")
    L.append("")
    L.append("The trajectory is NON-MONOTIC. room/garden peak at 5K-10K "
             "(room 0.078, garden 0.100) then dip at 14.5K (0.028/0.080) "
             "before rising again at 20K (0.120/0.105). bicycle peaks at 10K "
             "(0.153), dips at 14.5K (0.068), then RISES to 0.205 at 25K - "
             "its largest, deepest-complexity scene. Even so, NO scene "
             "reaches 25%: peaks are room 0.120, garden 0.105, bicycle 0.205. "
             "The 'EVERY scene' requirement fails primarily because room and "
             "garden are far below threshold; bicycle is closest but still "
             "4.5 points short. Because the gate is a hard >=25% in every "
             "scene and none reach it, exact active-pixel compaction has no "
             "structural headroom under this schedule.")
    L.append("")
    L.append("This is a genuine NO_STRUCTURAL_HEADROOM outcome at the frozen "
             "25K schedule, not a failed implementation. (Caveat: bicycle is "
             "trending up at 25K; a longer schedule might narrow the gap, but "
             "the spec fixes the schedule at 25K and forbids threshold "
             "shopping or schedule extension.)")
    L.append("")
    L.append("## Provenance")
    L.append("")
    L.append("- GPU: EPIC-05 A100 (%s)" % socket.gethostname())
    L.append("- Python %s" % platform.python_version())
    L.append("- Source lock: FasterGSFused ae665af2..., NeRFICG e458ac6... "
             "(frozen at /root/epic05_p16rb/src/nerficg)")
    L.append("- Predicate verified against kernels_backward.cuh (frozen source).")
    L.append("- All 3 scenes captured the full 7-checkpoint schedule "
             "(500/2K/5K/10K/14.5K/20K/25K), invariants all pass.")
    L.append("")
    L.append("## Artifacts")
    L.append("")
    for name in ["APBC0_ENVIRONMENT.json", "APBC0_SOURCE_LOCK.json",
                 "APBC0_REACHABILITY_TRACE.json", "APBC0_SUMMARY.json",
                 "APBC0_VERDICT.json", "APBC0_REPORT.md"]:
        p = ART / name
        if p.exists():
            L.append("- %s (sha256 %s...)" % (name, sha256(p)[:16]))
    L.append("")
    L.append("Phase B (metadata bench) and Phase C (kernel prototype) were "
             "NOT produced: the P1 stop rule forbids them without "
             "structural headroom.")

    (ART / "APBC0_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote APBC0_REPORT.md (%d lines)" % len(L))


if __name__ == "__main__":
    main()

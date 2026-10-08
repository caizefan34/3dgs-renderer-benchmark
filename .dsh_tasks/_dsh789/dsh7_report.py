#!/usr/bin/env python3
"""DSH-7 BARS0 report writer."""
import json
import hashlib
from pathlib import Path
import platform
import socket

ART = Path("/root/dsh_20261001/artifacts/DSH7_BARS0")


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    verdict = json.loads((ART / "BARS0_VERDICT.json").read_text())
    oracle = json.loads((ART / "BARS0_ADAPTIVE_ORACLE.json").read_text())
    tail = json.loads((ART / "BARS0_TAIL_PACKING.json").read_text())

    L = []
    L.append("# DSH-7 - BARS0: Bucket-Adaptive Replay Scheduling")
    L.append("")
    L.append("## Verdict")
    L.append("")
    L.append("**%s** (evidence %s)." % (verdict["decision"], verdict["evidence_level"]))
    L.append("")
    L.append("Phase A structural oracle shows large K-dependent headroom, but the "
             "frozen B1/B2 gates require MEASURED forward+backward renderer time "
             "(Phase B static-K microbench), which is not performed. The P0 is "
             "INVESTIGATE, not PROMOTE/NO_STRUCTURAL_HEADROOM.")
    L.append("")
    L.append("## Research question")
    L.append("")
    L.append("Taming 3DGS / Faster-GS use a fixed replay checkpoint spacing of 32 "
             "splats (forward stores color/transmittance every 32nd Gaussian; "
             "backward processes 32-splat buckets, one warp per bucket). Does a "
             "fixed K=32 leave meaningful exact systems headroom because per-tile "
             "depth complexity is highly heterogeneous?")
    L.append("")
    L.append("## Structural finding (Phase A + C oracle)")
    L.append("")
    L.append(oracle["structural_finding"])
    L.append("")
    L.append("Per-scene oracle reduction (bucket count and checkpoint bytes, "
             "K=32 -> K=64), at the latest mature iteration captured:")
    L.append("")
    L.append("| scene | iter | bucket-count reduction | checkpoint-bytes reduction |")
    L.append("|---|---|---|---|")
    for s, v in verdict["per_scene_oracle_reduction"].items():
        L.append("| %s | %d | %.1f%% | %.1f%% |" % (
            s, v["iter"], v["oracle_reduction"] * 100, v["K64_reduction"] * 100))
    L.append("")
    L.append("Mean oracle reduction: **%.1f%%**." % (verdict["mean_oracle_reduction"] * 100))
    L.append("")
    L.append("Because bucket count = ceil(S_t / K) is monotone non-increasing in "
             "K, the per-tile adaptive oracle selects K_t = 64 for EVERY tile. "
             "'Adaptive K' therefore collapses to global K=64; per-tile "
             "adaptivity adds no benefit over simply using K=64.")
    L.append("")
    L.append("## Phase D tail packing (K=32)")
    L.append("")
    L.append("Final-bucket valid-splat occupancy per tile at K=32 (latest iter):")
    L.append("")
    L.append("| scene | iter | frac final bucket <=8 | <=16 | <=24 | mean valid |")
    L.append("|---|---|---|---|---|---|")
    for s, d in tail["scenes"].items():
        pi = d["per_iter"]
        if pi:
            it = max(int(k) for k in pi.keys())
            v = pi[str(it)]
            L.append("| %s | %d | %.3f | %.3f | %.3f | %.1f |" % (
                s, it, v["frac_final_bucket_le8"], v["frac_final_bucket_le16"],
                v["frac_final_bucket_le24"], v["mean_final_bucket_valid"]))
    L.append("")
    L.append("~24% of final buckets have <=8 valid splats (idle lanes); ~50% have "
             "<=16. This is the short-tail waste Phase D targets, but it is an "
             "ORACLE only unless a gate passes.")
    L.append("")
    L.append("## Frozen gates status")
    L.append("")
    L.append("| gate | desc | status |")
    L.append("|---|---|---|")
    for k, g in verdict["gates"].items():
        L.append("| %s | %s | %s |" % (k, g["desc"], g["status"]))
    L.append("")
    L.append("## Why not PROMOTE (B1/B2 pending)")
    L.append("")
    L.append(verdict["reasoning"])
    L.append("")
    L.append("### Analytical note (not a measurement)")
    L.append("")
    L.append("The K-dependent costs are (a) forward checkpoint store "
             "(n_buckets x 256 x 16 bytes) and (b) backward checkpoint load + "
             "per-bucket launch overhead. Both scale with bucket count, so K=64 "
             "halves them. The K-INDEPENDENT costs (forward per-splat blending "
             "eval + backward per-splat gradient math) scale with total splats "
             "and are unchanged. Since per-splat eval typically dominates "
             "renderer time, the 46% bucket-count reduction likely translates to "
             "a SINGLE-DIGIT renderer-time fraction, below B1's 8% threshold. "
             "This is a hypothesis to be confirmed by Phase B measured timing, "
             "not a claim.")
    L.append("")
    L.append("## Phase B blocker (precise)")
    L.append("")
    L.append(verdict["phase_B_blocker"])
    L.append("")
    L.append("## Provenance")
    L.append("")
    L.append("- GPU: EPIC-05 A100 (%s)" % socket.gethostname())
    L.append("- Python %s" % platform.python_version())
    L.append("- Source lock: FasterGSFused ae665af2..., NeRFICG e458ac6... "
             "(frozen at /root/epic05_p16rb/src/nerficg)")
    L.append("- Predicate/structure verified against kernels_forward.cuh "
             "(extract_bucket_counts, blend_cu) and kernels_backward.cuh "
             "(blend_backward_cu).")
    L.append("")
    L.append("## Artifacts")
    L.append("")
    for name in ["BARS0_ENVIRONMENT.json", "BARS0_SOURCE_LOCK.json",
                 "BARS0_TILE_TRACE.json", "BARS0_STRUCTURAL_ORACLE.json",
                 "BARS0_ADAPTIVE_ORACLE.json", "BARS0_TAIL_PACKING.json",
                 "BARS0_VERDICT.json", "BARS0_SUMMARY.json", "BARS0_REPORT.md"]:
        p = ART / name
        if p.exists():
            L.append("- %s (sha256 %s...)" % (name, sha256_file(p)[:16]))
    L.append("")
    L.append("Absent (Phase B): BARS0_STATIC_K_BENCH.json - requires the "
             "K-parameterized kernel rebuild + CUDA-event timing harness.")

    (ART / "BARS0_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote BARS0_REPORT.md (%d lines)" % len(L))


if __name__ == "__main__":
    main()

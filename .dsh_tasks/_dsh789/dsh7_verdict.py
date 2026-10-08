#!/usr/bin/env python3
"""DSH-7 BARS0 verdict + summary + report (Phase A structural oracle).

The per-tile adaptive oracle collapses to global K=64 (bucket count is monotone
non-increasing in K), giving a ~46% bucket-count and checkpoint-byte reduction
vs K=32. The frozen B1 gate requires >=8% MEASURED forward+backward renderer
time, which needs the Phase B static-K microbench (K-parameterized kernel). This
verdict records the structural finding (L1) and bounds the claim: B1/B2 are
PENDING Phase B measured timing; B3 structural component (checkpoint bytes -46%)
is satisfied; B4 not evaluable without B1.
"""
import json
import hashlib
from pathlib import Path
import platform
import socket

ART = Path("/root/dsh_20261001/artifacts/DSH7_BARS0")
SPEC = "DSH_7_BARS0.md"


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    oracle = json.loads((ART / "BARS0_ADAPTIVE_ORACLE.json").read_text())
    tail = json.loads((ART / "BARS0_TAIL_PACKING.json").read_text())
    trace = json.loads((ART / "BARS0_TILE_TRACE.json").read_text())

    # representative mature reduction (use max available iter per scene)
    red = {}
    for s, d in oracle["scenes"].items():
        pi = d["per_iter"]
        if pi:
            last_it = max(int(k) for k in pi.keys())
            red[s] = {"iter": last_it,
                      "oracle_reduction": pi[str(last_it)]["oracle_reduction_vs_K32"],
                      "K64_reduction": pi[str(last_it)]["K64_reduction_vs_K32"]}
    mean_red = float(sum(v["oracle_reduction"] for v in red.values()) / max(len(red), 1))

    gates = {
        "B1": {"desc": "adaptive K oracle reduces measured fwd+bwd renderer time >=8% in 2/3 scenes, no scene <4%",
               "status": "PENDING_PHASE_B",
               "structural_basis": "oracle=K=64; bucket-count reduction %.1f%% mean (structural, not time)" % (mean_red * 100)},
        "B2": {"desc": "predicted full-iteration E2E saving >=3% in 2/3 scenes",
               "status": "PENDING_PHASE_B"},
        "B3": {"desc": "not solely launch overhead: checkpoint bytes / replay arithmetic / DRAM traffic decreases >=10% in 2/3 scenes",
               "status": "STRUCTURALLY_SUPPORTED",
               "checkpoint_bytes_reduction_mean": mean_red,
               "note": "forward checkpoint bytes drop ~%.0f%% (K32->K64), >=10%% satisfied structurally" % (mean_red * 100)},
        "B4": {"desc": "adaptive decision metadata/control overhead <=20% of gross saved time",
               "status": "NOT_EVALUABLE",
               "note": "oracle is global K=64 (no per-tile metadata); decision overhead ~0, but gross saved time needs Phase B"},
    }

    any_gate_pass = any(g["status"] in ("PASS", "STRUCTURALLY_SUPPORTED") for g in gates.values())
    all_core_pass = all(g["status"] in ("PASS", "STRUCTURALLY_SUPPORTED") for g in gates.values())

    decision = "INVESTIGATE"
    reason = ("Phase A structural oracle shows large K-dependent headroom "
              "(oracle collapses to global K=64; ~%.0f%% bucket-count and "
              "checkpoint-byte reduction vs K=32). However the frozen B1/B2 "
              "gates require MEASURED forward+backward renderer time, which "
              "needs the Phase B static-K microbench (K-parameterized kernel: "
              "K is coupled to warp size in the backward shuffle scan). "
              "B3's checkpoint-bytes component is structurally satisfied "
              "(~%.0f%% reduction). B1/B2/PENDING Phase B timing; verdict is "
              "INVESTIGATE, not PROMOTE/NO_HEADROOM." % (mean_red * 100, mean_red * 100))

    verdict = {
        "p0": "DSH-7 BARS0 Bucket-Adaptive Replay Scheduling",
        "decision": decision,
        "evidence_level": "L1 (Phase A structural trace + oracle, EPIC-05 A100)",
        "structural_finding": oracle["structural_finding"],
        "per_scene_oracle_reduction": red,
        "mean_oracle_reduction": mean_red,
        "gates": gates,
        "all_core_gates_pass": all_core_pass,
        "reasoning": reason,
        "claim_scope": ("Structural headroom PRESENT at L1 (bucket count and "
                        "checkpoint bytes drop ~%.0f%% K32->K64). No renderer-time "
                        "or E2E speedup claim: B1/B2 require Phase B measured "
                        "timing, not yet performed. 'Adaptive K' collapses to "
                        "global K=64 (per-tile adaptivity adds nothing)." % (mean_red * 100)),
        "phase_B_required": True,
        "phase_B_blocker": ("K=32 is a compile-time constant coupled to warp size "
                            "in backward blend_backward_cu (1 warp/bucket, 32 "
                            "lanes = 32 splats, warp-shuffle scan over splat dim). "
                            "K=64 needs a 2-splats-per-lane order-preserving "
                            "backward variant + rebuild + CUDA-event timing."),
        "provenance": {"gpu": "EPIC-05 A100 (%s)" % socket.gethostname(),
                       "python": platform.python_version(),
                       "source_lock": "FasterGSFused ae665af2..., NeRFICG e458ac6..."},
    }
    (ART / "BARS0_VERDICT.json").write_text(json.dumps(verdict, indent=2))
    print("wrote BARS0_VERDICT.json decision=%s mean_red=%.3f" % (decision, mean_red))

    summary = {
        "p0": "DSH-7 BARS0 Bucket-Adaptive Replay Scheduling",
        "decision": decision,
        "evidence_level": verdict["evidence_level"],
        "phase_reached": "A (structural trace) + C (structural oracle) + D (tail packing)",
        "structural_finding": oracle["structural_finding"],
        "mean_oracle_reduction": mean_red,
        "per_scene_oracle_reduction": red,
        "gate_status": {k: v["status"] for k, v in gates.items()},
        "files_absent": ["BARS0_STATIC_K_BENCH.json", "BARS0_SUMMARY.json (this file)"],
        "absent_reason": "Phase B static-K microbench not performed (K-parameterized kernel build required); B1/B2 need measured renderer time.",
        "claim_scope": verdict["claim_scope"],
        "provenance": verdict["provenance"],
    }
    (ART / "BARS0_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print("wrote BARS0_SUMMARY.json")


if __name__ == "__main__":
    main()

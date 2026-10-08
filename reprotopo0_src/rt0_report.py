#!/usr/bin/env python3
"""
ReproTopo0 report driver (packet Sec.7/Sec.10).

Aggregates the per-event parity / determinism / baseline / timing JSONs into the
10 required outputs + MANIFEST.sha256, then runs the FROZEN packet analyzer
(reprotopo0_analyze.py) to produce the mechanical verdict.
"""
import os
import sys
import json
import hashlib
import argparse
import subprocess

sys.path.insert(0, "/root/reprotopo0_src")
import rt0_common as C

FUSED_SSIM_SO = "/root/fused-ssim/build/lib.linux-x86_64-cpython-310/fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so"
ANALYZER = "/root/reprotopo0_src/reprotopo0_analyze.py"
PY = "/root/miniforge3/envs/gsplat/bin/python"

EVENTS = [("room", it) for it in (10100, 10200, 10300, 10400, 10500)] +          [("bicycle", it) for it in (10100, 10200, 10300, 10400, 10500)]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(datadir, sub, scene, it):
    p = os.path.join(datadir, sub, "%s_%d.json" % (scene, it))
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="/root/reprotopo0_s0")
    ap.add_argument("--outdir", default="/root/reprotopo0_s0/final")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dd = args.datadir

    recs = {}
    for scene, it in EVENTS:
        recs[(scene, it)] = {
            "parity": load(dd, "_parity", scene, it),
            "det": load(dd, "_det", scene, it),
            "baseline": load(dd, "_baseline", scene, it),
            "timing": load(dd, "_timing", scene, it),
        }
    missing = []
    for k, r in recs.items():
        for name, d in r.items():
            if d is None:
                missing.append("%s/%s_%d" % (name, k[0], k[1]))
    if missing:
        print("WARNING missing records:", missing)

    bin_hashes = {
        "c0_extension_sha256": sha256_file(C.C0_SO),
        "core_gsplat_cuda_sha256": sha256_file(C.CORE_SO),
        "fused_ssim_sha256": sha256_file(FUSED_SSIM_SO),
    }
    bin_match = all(bin_hashes[k] == C.AUTHORITATIVE[k] for k in bin_hashes)
    ckpt_hashes = {}
    ckpt_match = True
    for cell, info in C.CELLS.items():
        h = sha256_file(info["ckpt"])
        ckpt_hashes[cell] = h
        expected = C.AUTHORITATIVE["cells_4"].get(cell)
        if expected is None or h != expected:
            ckpt_match = False

    import torch
    gpus = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gpus.append({"index": i, "name": p.name,
                     "total_mem_gb": round(p.total_memory / 1e9, 2)})
    environment = {
        "packet": "REPROTOPO0_PACKET_20260930",
        "execution_target": "EPIC-05 A100 cluster (8x A100-SXM4-80GB)",
        "local_gpu_used": False,
        "hostname": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": sys.version.split()[0],
        "gpus": gpus,
        "binary_hashes": bin_hashes,
        "binary_hashes_match_packet": bin_match,
        "generated_at_utc": C.now_utc(),
    }
    C.emit_json(os.path.join(args.outdir, "REPROTOPO0_ENVIRONMENT.json"), environment)

    cfg = C.load_config()
    source_identity = {
        "authority": "N1 canonical trajectory + HM0 frozen C0 + DPC0 packet",
        "semantic_label": "REFERENCE_V1_ABSGRAD",
        "binary_pins": C.AUTHORITATIVE,
        "binary_hashes_verified": bin_hashes,
        "binary_hashes_match_packet": bin_match,
        "ckpt_hashes_verified": ckpt_hashes,
        "ckpt_hashes_match_packet": ckpt_match,
        "frozen_config": cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg),
        "reference_v1_dir": C.REF_V1_DIR,
        "rules": [
            "Full 64-hex identity checks before compute.",
            "Do not overwrite authoritative C0 binaries.",
            "No threshold relaxation after results.",
            "Analyzer owns verdict.",
            "BLOCKED is not DROP.",
        ],
    }
    C.emit_json(os.path.join(args.outdir, "REPROTOPO0_SOURCE_IDENTITY.json"), source_identity)

    cap_man_path = os.path.join(dd, "cap", "REPROTOPO0_CAPTURE_MANIFEST.json")
    with open(cap_man_path) as f:
        cap_man = json.load(f)
    snap_verify = {}
    for scene, it in EVENTS:
        cell = cap_man["cells"]["%s@10000" % scene]["events"][str(it)]
        sp = cell["snapshot_path"]
        snap_verify["%s_%d" % (scene, it)] = {
            "snapshot_sha256_recorded": cell["snapshot_sha256"],
            "snapshot_sha256_actual": sha256_file(sp) if os.path.exists(sp) else None,
        }
    cap_man["snapshot_hash_reverification"] = snap_verify
    cap_man["note"] = ("WORKLOAD-ONLY trace. NOT canonical continuation evidence. "
                       "Re-hashed at report time for integrity.")
    C.emit_json(os.path.join(args.outdir, "REPROTOPO0_CAPTURE_MANIFEST.json"), cap_man)

    baseline_per_event = []
    baseline_same_gpu_pass = True
    baseline_cross_pass = True
    baseline_perm_invariant_all = True
    for scene, it in EVENTS:
        b = recs[(scene, it)]["baseline"]
        if b is None:
            baseline_same_gpu_pass = False; baseline_cross_pass = False
            baseline_perm_invariant_all = False
            continue
        baseline_same_gpu_pass = baseline_same_gpu_pass and b["same_gpu_determinism_pass"]
        baseline_cross_pass = baseline_cross_pass and b["cross_a100_determinism_pass"]
        baseline_perm_invariant_all = baseline_perm_invariant_all and b["permutation_invariant"]
        baseline_per_event.append({
            "scene": scene, "iteration": it,
            "same_gpu_unique_hashes": b["same_gpu_unique_hashes"],
            "same_gpu_determinism_pass": b["same_gpu_determinism_pass"],
            "cross_unique_hashes": b["cross_unique_hashes"],
            "cross_a100_determinism_pass": b["cross_a100_determinism_pass"],
            "perm_unique_hashes": b["perm_unique_hashes"],
            "permutation_invariant": b["permutation_invariant"],
        })
    baseline_already_full_deterministic = baseline_same_gpu_pass and baseline_cross_pass
    baseline_audit = {
        "per_event": baseline_per_event,
        "aggregate": {
            "same_gpu_determinism_all": baseline_same_gpu_pass,
            "cross_a100_determinism_all": baseline_cross_pass,
            "original_permutation_invariant_all": baseline_perm_invariant_all,
        },
        "baseline_already_full_deterministic": baseline_already_full_deterministic,
        "note": ("Original event audited with identical RNG seed each run. "
                 "Baseline determinism does not stop development; ReproTopo must "
                 "add a distinct stronger property (permutation/tie invariance)."),
    }
    C.emit_json(os.path.join(args.outdir, "REPROTOPO0_BASELINE_AUDIT.json"), baseline_audit)

    parity_per_event = []
    policy_parity_pass = True
    for scene, it in EVENTS:
        p = recs[(scene, it)]["parity"]
        if p is None:
            policy_parity_pass = False; continue
        policy_parity_pass = policy_parity_pass and p["comparison"]["pass"]
        parity_per_event.append({
            "scene": scene, "iteration": it,
            "N_in": p["N_in"], "orig_N_out": p["orig_N_out"],
            "reprotopo_N_out": p["reprotopo_N_out"],
            "orig_clone": p["orig_actions"]["n_cloned"],
            "reprotopo_clone": p["reprotopo_meta"]["n_clone"],
            "orig_split": p["orig_actions"]["n_split"],
            "reprotopo_split": p["reprotopo_meta"]["n_split"],
            "orig_pruned_total": p["orig_actions"]["n_pruned_total"],
            "reprotopo_pruned_expanded": p["reprotopo_meta"]["n_prune_expanded"],
            "pass": p["comparison"]["pass"],
        })
    policy_parity = {
        "per_event": parity_per_event,
        "policy_parity_pass": policy_parity_pass,
        "note": ("Action-set/count parity (clone/split/prune sets, counts, final count). "
                "Child positions not compared (different RNG); full-state hashes expected to differ."),
    }
    C.emit_json(os.path.join(args.outdir, "REPROTOPO0_POLICY_PARITY.json"), policy_parity)

    det_per_event = []
    same_gpu_det_pass = True
    cross_det_pass = True
    perm_det_pass = True
    tie_det_pass = True
    total_nan = 0
    total_coll = 0
    for scene, it in EVENTS:
        d = recs[(scene, it)]["det"]
        if d is None:
            same_gpu_det_pass = False; cross_det_pass = False
            perm_det_pass = False; tie_det_pass = False; continue
        same_gpu_det_pass = same_gpu_det_pass and d["same_gpu_determinism_pass"]
        cross_det_pass = cross_det_pass and d["cross_a100_determinism_pass"]
        perm_det_pass = perm_det_pass and d["permutation_determinism_pass"]
        tie_det_pass = tie_det_pass and d["tie_stress_pass"]
        total_nan += d["nan_inf_count"]; total_coll += d["collision_count"]
        det_per_event.append({
            "scene": scene, "iteration": it,
            "same_gpu_unique_hashes": d["same_gpu_unique_hashes"],
            "cross_unique_hashes": d["cross_unique_hashes"],
            "perm_unique_hashes": d["perm_unique_hashes"],
            "perm_all_equal_to_base": d["perm_all_equal_to_base"],
            "tie_rows": d["tie_rows"], "tie_unique_hashes": d["tie_unique_hashes"],
            "collision_count": d["collision_count"], "nan_inf_count": d["nan_inf_count"],
        })
    determinism = {
        "per_event": det_per_event,
        "aggregate": {
            "same_gpu_determinism_pass": same_gpu_det_pass,
            "cross_a100_determinism_pass": cross_det_pass,
            "permutation_determinism_pass": perm_det_pass,
            "tie_stress_pass": tie_det_pass,
            "total_nan_inf_count": total_nan,
            "total_id_collision_count": total_coll,
        },
    }
    C.emit_json(os.path.join(args.outdir, "REPROTOPO0_DETERMINISM.json"), determinism)

    timing_per_event = []
    max_ratio = 0.0
    max_amort = 0.0
    max_vram = 0.0
    sum_nan = 0
    sum_coll = 0
    for scene, it in EVENTS:
        t = recs[(scene, it)]["timing"]
        if t is None:
            max_ratio = 99.0; max_amort = 99.0; max_vram = 99.0; continue
        max_ratio = max(max_ratio, t["max_event_wall_ratio"])
        max_amort = max(max_amort, t["amortized_full_training_overhead_fraction"])
        max_vram = max(max_vram, t["extra_vram_fraction"])
        sum_nan += t["nan_inf_count"]; sum_coll += t["id_collision_count"]
        timing_per_event.append({
            "scene": scene, "iteration": it,
            "orig_event_wall_s": t["orig_event_wall_s"],
            "reprotopo_event_wall_s": t["reprotopo_event_wall_s"],
            "max_event_wall_ratio": t["max_event_wall_ratio"],
            "amortized_full_training_overhead_fraction": t["amortized_full_training_overhead_fraction"],
            "extra_vram_fraction": t["extra_vram_fraction"],
            "per_iter_wall_mean_s": t["per_iter_wall_mean_s"],
            "orig_peak_bytes": t["orig_peak_bytes"],
            "reprotopo_peak_bytes": t["reprotopo_peak_bytes"],
        })
    timing = {
        "per_event": timing_per_event,
        "aggregate": {
            "max_event_wall_ratio": max_ratio,
            "max_amortized_full_training_overhead_fraction": max_amort,
            "max_extra_vram_fraction": max_vram,
            "total_nan_inf_count": sum_nan,
            "total_id_collision_count": sum_coll,
            "repeats_per_event": 256,
        },
    }
    C.emit_json(os.path.join(args.outdir, "REPROTOPO0_TIMING.json"), timing)

    candidate_adds_stronger_property = (
        same_gpu_det_pass and cross_det_pass and perm_det_pass and tie_det_pass
        and (not baseline_perm_invariant_all))
    summary = {
        "baseline_already_full_deterministic": baseline_already_full_deterministic,
        "candidate_adds_stronger_property": candidate_adds_stronger_property,
        "policy_parity_pass": policy_parity_pass,
        "same_gpu_determinism_pass": same_gpu_det_pass,
        "permutation_determinism_pass": perm_det_pass,
        "cross_a100_determinism_pass": cross_det_pass,
        "tie_stress_pass": tie_det_pass,
        "global": {
            "max_event_wall_ratio": max_ratio,
            "amortized_full_training_overhead_fraction": max_amort,
            "extra_vram_fraction": max_vram,
            "nan_inf_count": sum_nan,
            "id_collision_count": sum_coll,
        },
    }
    summary_path = os.path.join(args.outdir, "REPROTOPO0_SUMMARY.json")
    C.emit_json(summary_path, summary)

    subprocess.run([PY, ANALYZER, summary_path], check=True)
    verdict_path = os.path.join(args.outdir, "REPROTOPO0_VERDICT.json")
    with open(verdict_path) as f:
        verdict = json.load(f)

    report = build_report(environment, summary, verdict, policy_parity,
                          determinism, baseline_audit, timing, bin_match, ckpt_match)
    with open(os.path.join(args.outdir, "REPROTOPO0_REPORT.md"), "w") as f:
        f.write(report)

    manifest_files = [
        "REPROTOPO0_ENVIRONMENT.json", "REPROTOPO0_SOURCE_IDENTITY.json",
        "REPROTOPO0_CAPTURE_MANIFEST.json", "REPROTOPO0_BASELINE_AUDIT.json",
        "REPROTOPO0_POLICY_PARITY.json", "REPROTOPO0_DETERMINISM.json",
        "REPROTOPO0_TIMING.json", "REPROTOPO0_SUMMARY.json",
        "REPROTOPO0_VERDICT.json", "REPROTOPO0_REPORT.md",
    ]
    lines = []
    for fn in sorted(manifest_files):
        h = sha256_file(os.path.join(args.outdir, fn))
        lines.append("%s  %s" % (h, fn))
    with open(os.path.join(args.outdir, "MANIFEST.sha256"), "w") as f:
        f.write(chr(10).join(lines) + chr(10))

    print("WROTE outputs to", args.outdir)
    print("VERDICT:", json.dumps(verdict))
    print("bin_match:", bin_match, "ckpt_match:", ckpt_match)


def build_report(env, summary, verdict, parity, det, baseline, timing, bin_match, ckpt_match):
    fence = chr(96) * 3
    L = []
    L.append("# ReproTopo0 — Deterministic Topology State Machine (FROZEN)")
    L.append("")
    L.append("Packet: REPROTOPO0_PACKET_20260930  ")
    L.append("Generated: %s  " % env["generated_at_utc"])
    L.append("Execution: %s  " % env["execution_target"])
    L.append("Local GPU used: %s  " % env["local_gpu_used"])
    L.append("")
    L.append("## Mechanical verdict")
    L.append("")
    L.append(fence + "json")
    L.append(json.dumps(verdict, indent=2))
    L.append(fence)
    L.append("")
    L.append("## Binary / checkpoint identity")
    L.append("")
    L.append("- C0/core/fused_ssim hashes match packet pins: **%s**" % bin_match)
    L.append("- Authoritative ckpt hashes match: **%s**" % ckpt_match)
    L.append("")
    L.append("## Policy parity (Sec.4)")
    L.append("")
    L.append("policy_parity_pass = **%s**" % summary["policy_parity_pass"])
    L.append("")
    L.append("| scene | iter | N_in | orig_N_out | rt_N_out | clone o/r | split o/r | pruned o/r | pass |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for e in parity["per_event"]:
        L.append("| %s | %d | %d | %d | %d | %d/%d | %d/%d | %d/%d | %s |" % (
            e["scene"], e["iteration"], e["N_in"], e["orig_N_out"], e["reprotopo_N_out"],
            e["orig_clone"], e["reprotopo_clone"], e["orig_split"], e["reprotopo_split"],
            e["orig_pruned_total"], e["reprotopo_pruned_expanded"], e["pass"]))
    L.append("")
    L.append("## Determinism stress (Sec.5)")
    L.append("")
    a = det["aggregate"]
    L.append("- same-GPU (64x) deterministic: **%s**" % a["same_gpu_determinism_pass"])
    L.append("- cross-A100 (GPU0-3) identical: **%s**" % a["cross_a100_determinism_pass"])
    L.append("- permutation (8x) invariant: **%s**" % a["permutation_determinism_pass"])
    L.append("- exact-tie stress deterministic: **%s**" % a["tie_stress_pass"])
    L.append("- total NaN/Inf: %d ; total ID collisions: %d" % (a["total_nan_inf_count"], a["total_id_collision_count"]))
    L.append("")
    L.append("## Baseline audit (Sec.2)")
    L.append("")
    ba = baseline["aggregate"]
    L.append("- baseline already full-deterministic (same-GPU + cross-A100): **%s**" % baseline["baseline_already_full_deterministic"])
    L.append("- original permutation-invariant: **%s**" % ba["original_permutation_invariant_all"])
    L.append("- candidate adds stronger property (permutation/tie invariance): **%s**" % summary["candidate_adds_stronger_property"])
    L.append("")
    L.append("## Timing (Sec.6)")
    L.append("")
    g = summary["global"]
    L.append("- max event wall ratio (reprotopo/original): **%.4f**" % g["max_event_wall_ratio"])
    L.append("- amortized full-training overhead fraction: **%.6f**" % g["amortized_full_training_overhead_fraction"])
    L.append("- extra peak VRAM fraction: **%.4f**" % g["extra_vram_fraction"])
    L.append("")
    L.append("## Boundary")
    L.append("")
    L.append("This is NOT a new density-control algorithm. It makes topology-changing "
             "3DGS training a reproducible state machine once deterministic gradients "
             "are available. The captured trace is workload-only, not canonical "
             "continuation evidence.")
    L.append("")
    return chr(10).join(L)


if __name__ == "__main__":
    main()

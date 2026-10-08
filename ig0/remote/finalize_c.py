import json, os, hashlib, platform, torch
OUT = "/root/ig0_run/out"
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

summary = json.load(open(os.path.join(OUT, "IG0_SUMMARY.json")))
verdict = json.load(open(os.path.join(OUT, "IG0_VERDICT.json")))
env = json.load(open(os.path.join(OUT, "IG0_ENVIRONMENT.json")))
ident = json.load(open(os.path.join(OUT, "IG0_SOURCE_IDENTITY.json")))
rp = json.load(open(os.path.join(OUT, "IG0_RADIUS_PARITY.json")))

cells = ["room@10000", "room@15000", "bicycle@10000", "bicycle@15000"]
L = []
L.append("# InfoGeo-GS IG0 GPU Report (EPIC-05 A100)")
L.append("")
L.append("## STATUS")
L.append("```")
L.append(verdict["verdict"])
L.append("```")
L.append("selected_fixed_variant = %s ; strong=%s ; promote=%s" % (verdict["selected_fixed_variant"], verdict["strong_variants"], verdict["promote_variants"]))
L.append("ig0_analyze.py owns verdict; manual override = false.")
L.append("")
L.append("## EXECUTION")
L.append("```")
L.append("gpu_execution_host: %s" % env["gpu_execution_host"])
L.append("GPU: %s" % env["GPU_model"])
L.append("CUDA_VISIBLE_DEVICES: %s" % env["CUDA_VISIBLE_DEVICES"])
L.append("torch: %s ; CUDA: %s ; python: %s" % (env["torch"], env["CUDA_version"], env["python"]))
L.append("remote/non-local confirmation: %s" % env["remote_non_local_confirmation"])
L.append("```")
L.append("")
L.append("## IDENTITY")
L.append("```")
L.append("C0 extension sha256: %s" % ident["c0_extension_sha256_live"])
L.append("core gsplat_cuda.so sha256: %s" % ident["core_gsplat_cuda_so_sha256_live"])
L.append("fused_ssim_cuda.so sha256: %s" % ident["fused_ssim_cuda_so_sha256_live"])
L.append("prereg_literal_valid=%s authoritative_live_hash_match=%s" % (ident["prereg_literal_valid"], ident["authoritative_live_hash_match"]))
L.append("trainer: %s" % ident["trainer"])
L.append("checkpoints (iter/arm/N/has_optimizer_state):")
for c, ci in ident["checkpoints"].items():
    L.append("  %s: iter=%d arm=%s N=%d opt=%s" % (c, ci["iteration"], ci["arm"], ci["n_gaussians"], ci["has_optimizer_state"]))
L.append("```")
L.append("")
L.append("## CORRECTNESS")
L.append("CPU reference (EPIC-05, fp64): Fisher PSD / closed-form-vs-differential / quaternion Jacobian / FR-GLOBAL & FR-LOCAL radius parity / NaN — 6/6 PASS.")
L.append("GPU radius parity (real native Adam displacement, per probe): worst FR-GLOBAL rel = %.2e; worst FR-LOCAL median = %.2e; worst FR-LOCAL p99 = %.2e (tolerances 1e-5 / 1e-5 / 1e-4). NaN/Inf = 0 across all 32 probes." % (
    max(rp["aggregates"][c]["fr_global_radius_rel_worst"] for c in cells),
    max(rp["aggregates"][c]["fr_local_radius_median_worst"] for c in cells),
    max(rp["aggregates"][c]["fr_local_radius_p99_worst"] for c in cells)))
L.append("")
L.append("## PRIMARY RESULTS (4 cells x 2 variants)")
L.append("| cell | variant | valid | R_src_median | R_src_p10 | peer_win | holdout_win | holdout_PSNR_med | holdout_PSNR_worst | correct |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for c in cells:
    for v in ["FR-GLOBAL", "FR-LOCAL"]:
        m = summary["cells"][c]["variants"][v]
        L.append("| %s | %s | %d/8 | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %s |" % (
            c, v, m["valid_source_descent_count"], m["R_src_median"], m["R_src_p10"],
            m["peer_win_rate"], m["truehold_win_rate"], m["truehold_psnr_delta_median_db"],
            m["truehold_psnr_delta_worst_db"], m["correctness_pass"]))
L.append("")
L.append("PROMOTE thresholds (per cell): valid>=6, R_src_median>=1.15, R_src_p10>=0.80, peer_win>=0.55, holdout_win>=0.55, holdout_PSNR_med>=-0.02, worst>=-0.10.")
L.append("STRONG thresholds: valid>=7, R_src_median>=1.30, R_src_p10>=1.00, peer_win>=0.65, holdout_win>=0.60, PSNR_med>=0.00, worst>=-0.05.")
L.append("No cell meets even PROMOTE for either variant (R_src_median 0.41-0.87 < 1.15; peer_win <= 0.125 < 0.55; room holdout_win = 0.0).")
L.append("")
L.append("## INTERPRETATION")
L.append("OBSERVED: With geometry direction from the intrinsic Fisher-Rao metric, magnitude-matched to native Adam's Fisher trust radius, the one-step source loss descent is 0.41x-0.87x of Adam's (R_src_median), and cross-view (peer) and true-holdout mean losses are not lower than Adam's (win rate 0.0-0.75, mostly <=0.125).")
L.append("EVIDENCE: 32 source probes (4 cells x 8), valid source descent 8/8 per cell, correctness PASS, NaN/Inf=0; frozen analyzer verdict IG0_DROP.")
L.append("CONCLUSION: The Fisher-Rao matched-radius geometry direction does not improve over native Adam for standard photometric 3DGS geometry updates; it is worse on source and does not help cross-view/holdout. InfoGeo-GS IG0 CLOSED (no rescue per prereg section 12/19).")
L.append("")
L.append("## ARTIFACTS")
L.append("remote: /root/ig0_run/out/ ; local mirror: C:\\Users\\36570\\3dgs-renderer-benchmark\\ig0\\ig0_gpu_run\\ ; manifest: MANIFEST.sha256 (self-entry excluded).")
L.append("knowledge note: D:\\学习\\3DGS渲染器对比\\笔记 - InfoGeo-GS IG0 Fisher-Rao几何方向Oracle 2026-09-30.md")
with open(os.path.join(OUT, "IG0_REPORT.md"), "w") as f:
    f.write("\n".join(L) + "\n")

# ---- manifest (sha256  filename), self excluded ----
targets = []
for nm in sorted(os.listdir(OUT)):
    if nm == "MANIFEST.sha256":
        continue
    targets.append(os.path.join(OUT, nm))
for nm in ["ig0_fisher.py", "ig0_reference_checks.py", "ig0_fisher_torch.py",
           "ig0_gpu_probe.py", "ig0_analyze.py", "full_grid.log"]:
    p = os.path.join("/root/ig0_run", nm)
    if os.path.exists(p):
        targets.append(p)
lines = []
for p in targets:
    rel = os.path.basename(p)
    lines.append("%s  %s" % (sha256_file(p), rel))
lines.append("# self-entry excluded: MANIFEST.sha256")
with open(os.path.join(OUT, "MANIFEST.sha256"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("report + manifest written; entries:", len(targets))

#!/usr/bin/env python3
"""Package the completed R3 hard-stop evidence; no renderer execution or timing."""
import hashlib, json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "higs-p2-1a-r3"
RAW = json.load(open(ROOT / "artifacts" / "higs-p2-1a-r3-forward-raw.json"))
TRACE = json.load(open(ROOT / "artifacts" / "higs-p2-1a-r3-trace-pre.json"))
PATCH = OUT / "higs-p2-1a-r3-semantic-repair.patch"
SO = OUT / "gsplat_cuda_p2_1a_r3.so"

def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")

def main():
    patch_sha = hashlib.sha256(PATCH.read_bytes()).hexdigest(); so_sha = hashlib.sha256(SO.read_bytes()).hexdigest()
    trace = TRACE["room"]; rows = trace["entries"][:20]
    weight_deltas = [abs(x["r2"]["actual_native_weight_one_hot"] - x["base"]["weight"]) for x in rows]
    first = next((x for x in rows if abs(x["r2"]["actual_native_weight_one_hot"] - x["base"]["weight"]) > 1e-5), None)
    provenance = {"round": "P2-1A-R3", "date": str(date.today()), "base": "frozen C0 V3 F9 + SCALAR_ADJOINT + H8-MR flat forward", "test_parent_r2": {"so_sha256": "d7cb5d83721473de3729b7baf4500592427e6eae31d0bc2b20fce126a74debae", "patch_sha256": "0d5434fc2e943bcc0310b72bdb8368347a42136296619ec2fa2c9e1cab9c85b8"}, "r3": {"so": SO.name, "so_sha256": so_sha, "patch": PATCH.name, "patch_sha256": patch_sha}, "timing_run": False, "fp16_on_test_path": False, "source_files_changed": ["remote_work/native_adapt/MacroTileRasterize.cu"]}
    conic = {"f9_representation": "(a,b,c) = symmetric inverse covariance upper triangle", "base_quadratic_form": "q_BASE = a*dx^2 + 2*b*dx*dy + c*dy^2", "r3_quadratic_form": "q_TEST = (l0*dx + l1*dy)^2 + (l2*dy)^2, l0=sqrt(a), l1=b/l0, l2=sqrt(c-l1^2)", "mapping": "Cholesky reconstruction for radii-valid projected rows", "equivalent_for_valid_f9": True, "non_spd_valid_rows": {s: RAW["scenes"][s]["quadratic"]["n_non_spd_valid"] for s in RAW["scenes"]}, "finding": "The prior non-SPD rates came from uninitialized conic storage in radii==0 culled rows and are not valid F9 renderer inputs."}
    alpha = {"base": {"sigma": "0.5*q", "alpha": "min(0.999, opacity*exp(-sigma))", "reject": "sigma < 0 or alpha < 1/255", "terminal": "next_T=T*(1-alpha); if next_T<=1e-4, stop without that contributor"}, "r3": {"sigma_log2": "0.5*log2(e)*q - log2(opacity)", "alpha": "min(0.999, exp2(-sigma_log2))", "reject": "sigma_log2 >= log2(255)", "terminal": "R3 suppresses terminal RGB contribution and marks remaining work inactive"}, "difference": "The alpha cap and terminal contribution were repaired, but they did not repair the observed per-splat weight mismatch."}
    support = {"base_predicate": "AccuTile ellipse/fine-tile intersection for q <= min(3.33^2, 2*ln(opacity*255))", "r3_predicate": "macro AccuTile ellipse enumeration followed by native fine-mask center_hit || edge_hit", "macro_compression": "allowed", "effective_fine_tile_gate": {"pass": False, "missing": "not established as zero", "extra": "not established as zero", "semantic_duplicates": "not established as zero"}, "evidence": "One-hot actual-raster traces show the native per-tile queue does not assign BASE's first contributors their expected weights despite equal reconstructed q/alpha."}
    quad = {s: RAW["scenes"][s]["quadratic"] for s in RAW["scenes"]}; quad["gate"] = "PASS for all radii-valid F9 rows sampled; FP32 reassociation envelope"
    structural = {"gate": "FAIL", "reason": "actual native queue support/weight trace diverges; exact effective fine-tile missing/extra/duplicate equality was not demonstrated", "room_trace_tile": trace["tile"], "room_trace_pixel": trace["pixel"], "n_gids": trace["n_gids"]}
    ordering = {"gate": "PASS", "non_tie_inversions": 0, "tie_only": 0, "reason": "R3 does not alter the segmented depth sort; frozen R2 order audit remains applicable"}
    weights = {"gate": "FAIL", "diagnostic": {"scene": "room", "tile": trace["tile"], "pixel": trace["pixel"], "n_gids": trace["n_gids"], "traced": len(rows)}, "first_divergence": {"quantity": "actual native per-splat weight / effective queue support", "gid": first["gid"], "base_weight": first["base"]["weight"], "r3_actual_one_hot_weight": first["r2"]["actual_native_weight_one_hot"]}, "mismatched_traced_splats": sum(d > 1e-5 for d in weight_deltas), "max_abs_weight_error": max(weight_deltas), "rule": "q, reconstructed alpha, and pre-queue algebra agree before this point; the actual one-hot queue result diverges."}
    final = {"verdict": "P2_1A_NATIVE_HIERARCHY_DROP", "hard_stop": True, "reason": "R3 retains a large per-splat weight/effective-support mismatch and RGB is outside the exact FP32 envelope.", "gates": {"A_quadratic": "PASS", "B_support": "FAIL", "C_ordering": "PASS", "D_per_splat_weight": "FAIL", "E_rgb_alpha": "FAIL"}, "timing_authorized": False, "next_action": "Drop P2-1A native hierarchy; archive R3 evidence. Do not create R4 or run authoritative timing."}
    dump("provenance.json", provenance); dump("per_splat_trace.json", TRACE); dump("conic_semantics.json", conic); dump("alpha_semantics.json", alpha); dump("support_semantics.json", support); dump("quadratic_equivalence.json", quad); dump("structural_equivalence.json", structural); dump("ordering_equivalence.json", ordering); dump("weight_equivalence.json", weights); dump("forward_correctness.json", RAW); dump("final_gate.json", final)
    report = f"""# P2-1A-R3 — Final Semantic Correctness Repair\n\n**Verdict:** `P2_1A_NATIVE_HIERARCHY_DROP`\n\nR3 kept the prescribed F9 → macro partition → segmented macro sort → 32-G masks → active queue → native raster → post-compose path. No benchmark was run.\n\n## Forensic result\n\nThe first divergent quantity is the actual native per-splat weight/effective queue support. On room tile {trace['tile']}, pixel {trace['pixel']}, the first traced gid {first['gid']} has BASE weight `{first['base']['weight']:.9g}` and R3 one-hot native weight `{first['r2']['actual_native_weight_one_hot']:.9g}`. The first 20 traces contain {sum(d > 1e-5 for d in weight_deltas)} mismatches. q and reconstructed alpha agree before the queue.\n\nF9 conics are inverse-conic coefficients: `q = a*dx² + 2*b*dx*dy + c*dy²`. R3's Cholesky form reconstructs this within the FP32 reassociation envelope for all radii-valid samples. The prior non-SPD classification included uninitialized values of already culled rows.\n\nR3 aligned the alpha cap and terminal contributor rule, but full-frame RGB remains off: room `{RAW['scenes']['room']['rgb']['rel_l2']:.6g}`, bicycle `{RAW['scenes']['bicycle']['rgb']['rel_l2']:.6g}`, garden `{RAW['scenes']['garden']['rgb']['rel_l2']:.6g}`.\n\nNo further rescue is authorized. Timing is not authorized.\n"""
    (ROOT / "reports" / "higs" / "p2-1a-r3-semantic-repair.md").write_text(report, encoding="utf-8")

if __name__ == "__main__": main()

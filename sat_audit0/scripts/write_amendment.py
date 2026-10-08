import json, os, datetime, hashlib

ROOT = "/mnt/storage_pool/liaoyuanjun/fused_profile0"
SYN = os.path.join(ROOT, "SAT_AUDIT0")
SATSYN = os.path.join(ROOT, "SATURATION_SYNTHESIS")
NOW = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
def load(p):
    with open(p) as f: return json.load(f)
def dump(path, obj):
    with open(path, "w") as f: json.dump(obj, f, indent=2)

vd = load(f"{SYN}/SAT_AUDIT0_VERDICT.json")
ch = load(f"{SYN}/SAT_AUDIT0_CORRECTED_HEADROOM.json")
sc = load(f"{SYN}/SAT_AUDIT0_SCENE_COMPARISON.json")

# Amendment: DO NOT overwrite original SATURATION_SYNTHESIS evidence (section 18)
amendment = {
  "schema": "SATURATION_SYNTHESIS_AMENDMENT_V1",
  "generated_utc": NOW,
  "amendment_to": "SATURATION_SYNTHESIS (FUSED_PROFILE0 negative-paper backbone)",
  "reason": "SAT-AUDIT0 (objective c00ad9f5) found a measurement artifact in the FUSED_PROFILE0 profile that affects the Q1 component breakdown and the 'bicycle/garden framework-bound/saturated' reading.",
  "original_evidence_preserved": "SATURATION_SYNTHESIS/*.json and FUSED_PROFILE0_*/AABB_PROD0_* are NOT modified. This amendment supersedes only the INTERPRETATION of the non-kernel time.",
  "audit_verdict": vd["primary_classification"],
  "artifact": vd["artifact_found"],
  "what_changed": {
    "original_claim": "bicycle/garden are framework/CPU-bound (f_raster 0.042/0.023), only room raster-bound; baseline saturated for large scenes.",
    "corrected_claim": "The ~275-288ms non-kernel time is per-iteration image decoding (data loading) inside the FP1/FP2 timed region (PRELOADING_LEVEL=0). Excluding it, ALL scenes are raster-dominant (corrected f_raster room 0.894, bicycle 0.936, garden 0.880).",
    "unchanged": "The exact per-tile AABB reject is intrinsic to FasterGSFused (C32==B32, S_reject=1.0, max_abs=0). The AABB-PROD0 production S_prod=0.584 result and DROP_MICROBENCH_ONLY verdict stand; the corrected profile changes the Q1 denominator, not the reject-intrinsic fact.",
  },
  "corrected_f_raster": {s: ch["per_scene"][s]["corrected_f_raster"] for s in ["room","bicycle","garden"]},
  "original_f_raster": {s: ch["per_scene"][s]["original_f_raster_FP2"] for s in ["room","bicycle","garden"]},
  "corrected_T_iteration_ms": {s: sc["per_scene"][s]["T_iteration_corrected_preloadedPL2_ms"] for s in ["room","bicycle","garden"]},
  "no_reopen_statement": "Per objective section 21, corrected substantial raster headroom is REPORTED but does NOT automatically reopen AABB-PROD0 or any optimization lane (CGT/DPS/BES/ACG/AMPS/RetireGS/ClockGS remain CLOSED). Return to PI for direction.",
  "evidence_package": "fused_profile0/SAT_AUDIT0/ (SAT_AUDIT0_VERDICT.json, REPORT.md, and the mode JSONs).",
}
dump(os.path.join(SATSYN, "AMENDMENT_20261006_SAT_AUDIT0.json"), amendment)
print("WROTE SATURATION_SYNTHESIS/AMENDMENT_20261006_SAT_AUDIT0.json")
print(json.dumps(amendment["corrected_f_raster"], indent=0))
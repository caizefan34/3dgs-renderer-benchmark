#!/usr/bin/env python3
import json, sys

CELLS=("room@10000","room@15000","bicycle@10000","bicycle@15000")
d=json.load(open(sys.argv[1]))

if not d.get("source_identity_pass", False):
    out={"verdict":"DPC0_SOURCE_IDENTITY_FAIL"}
elif not d.get("build_integrity_pass", False):
    out={"verdict":"DPC0_BUILD_INTEGRITY_FAIL"}
elif not d.get("same_gpu_determinism_pass", False):
    out={"verdict":"DPC0_NONDETERMINISTIC"}
elif not d.get("gradient_correctness_pass", False) or not d.get("one_step_equivalence_pass", False):
    out={"verdict":"DPC0_GRADIENT_INCORRECT"}
else:
    def promote(x):
        return (
            x["same_gpu_deterministic"] and
            x["ensemble_cos_median"] >= .999999 and
            x["ensemble_cos_p10"] >= .999995 and
            x["ensemble_rel_l2_median"] <= 5e-5 and
            x["ensemble_rel_l2_p90"] <= 1e-4 and
            x["one_step_equivalence_pass"] and
            x["backward_speedup"] >= 1.35 and
            x["peak_extra_vram_fraction"] <= .40 and
            x["nan_inf_count"] == 0
        )
    def strong(x):
        return (
            promote(x) and
            x["cross_a100_bitwise_equal"] and
            x["directional_conclusive"] and
            x["directional_rel_err_median"] <= 5e-3 and
            x["directional_rel_err_p90"] <= 2e-2 and
            x["backward_speedup"] >= 1.75 and
            x["peak_extra_vram_fraction"] <= .25
        )

    xs=[d["cells"][c] for c in CELLS]
    all_promote=all(promote(x) for x in xs)
    all_strong=all(strong(x) for x in xs)
    median_speed=float(d["global"]["backward_speedup_median"])

    if all_strong and median_speed >= 2.0:
        v="DPC0_STRONG_PROMOTE"
    elif all_promote and median_speed >= 1.5:
        v="DPC0_PROMOTE"
    else:
        v="DPC0_TOO_SLOW"
    out={
        "verdict":v,
        "all_promote_cells":all_promote,
        "all_strong_cells":all_strong,
        "backward_speedup_median":median_speed
    }

open(sys.argv[1].replace("SUMMARY","VERDICT"),"w").write(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))

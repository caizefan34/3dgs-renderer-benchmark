import json, numpy as np
import os
out = {}
root = "results/phase-c30"
for (fn, label) in [("c30_a_param_frequency.json","A"),
    ("c30_b_birth_warmstart.json","B"),("c30_c_renderer_optimizer.json","C"),
    ("c30_d_regime.json","D"),("c30_e_loss_frequency.json","E"),
    ("c30_f_cross_iteration.json","F"),("c30_g_multigpu.json","G"),
    ("c30_h_precision.json","H")]:
    d = json.load(open(os.path.join(root,fn)))
    out[label] = {}
    if label == "A":
        out[label]["verdict"] = d.get("verdict","?")
        out[label]["spread"] = d.get("spread",0)
        pg = d.get("param_group_stats",{})
        for g in pg:
            e = pg[g]["early"]
            l = pg[g]["late"]
            out[label][g] = {"early_grad": e["mean_grad_norm"], "late_grad": l["mean_grad_norm"],
                "early_update": e["mean_update_norm"], "late_update": l["mean_update_norm"],
                "early_rel": e["mean_rel_change"], "late_rel": l["mean_rel_change"]}
    elif label == "B":
        out[label]["n_birth_events"] = d.get("n_birth_events",0)
        events = d.get("birth_events",[])
        out[label]["births"] = [{"step": e["step"], "n_new": e["n_new"], "post_steps": len(e.get("post_tracking",[]))} for e in events[:10]]
    elif label == "C":
        out[label]["correlations"] = d.get("correlations",{})
    elif label == "D":
        out[label]["n_regimes"] = d.get("n_regimes",0)
        out[label]["regime_summary"] = d.get("regime_summary",{})
    elif label == "E":
        out[label]["component_timing"] = d.get("component_timing_ms",{})
        out[label]["temporal_corr"] = d.get("temporal_corr",{})
    elif label == "F":
        out[label]["cross_iteration_corr"] = d.get("cross_iteration_corr",{})
    elif label == "G":
        out[label]["profile"] = d.get("profile",{})
        out[label]["scaling_estimate"] = d.get("scaling_estimate",{})
    elif label == "H":
        for prec in ["fp32","tf32","mp"]:
            if prec in d:
                out[label][prec] = d[prec]
        for k in d:
            if "vs_fp32" in k:
                out[label][k] = d[k]
json.dump(out, open("results/phase-c30/c30_summary.json","w"), indent=2)
print("Summary saved")

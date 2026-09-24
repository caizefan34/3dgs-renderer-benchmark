#!/usr/bin/env python3
"""Quick extraction of C30 summary from json files on mx."""
import json, os
root = "/home/liaoyuanjun/3dgs-renderer-benchmark/results/phase-c30"
out = {}
for fn, label in [("c30_a_param_frequency.json","A"),("c30_b_birth_warmstart.json","B"),
    ("c30_c_renderer_optimizer.json","C"),("c30_d_regime.json","D"),
    ("c30_e_loss_frequency.json","E"),("c30_f_cross_iteration.json","F"),
    ("c30_g_multigpu.json","G"),("c30_h_precision.json","H")]:
    d = json.load(open(os.path.join(root,fn)))
    out[label] = d
    if label == "A":
        out[label] = {"verdict":d.get("verdict"),"n_steps":d.get("n_steps"),"spread":d.get("spread"),
            "param_group_stats":d.get("param_group_stats")}
    elif label == "B":
        out[label] = {"n_birth_events":d.get("n_birth_events"),"n_steps":d.get("n_steps")}
        if d.get("birth_events"):
            be = d["birth_events"]
            out[label]["birth_events"] = [{"step":e["step"],"n_new":e["n_new"],"n_post":len(e.get("post_tracking",[])),
                "post_tracking":e.get("post_tracking",[])[:20]} for e in be]
    elif label == "C":
        out[label] = {"n_steps":d.get("n_steps"),"correlations":d.get("correlations")}
    elif label == "D":
        out[label] = {"n_steps":d.get("n_steps"),"n_regimes":d.get("n_regimes"),"regime_summary":d.get("regime_summary")}
    elif label == "E":
        out[label] = {"n_steps":d.get("n_steps"),"component_timing_ms":d.get("component_timing_ms"),"temporal_corr":d.get("temporal_corr")}
    elif label == "F":
        out[label] = {"n_steps":d.get("n_steps"),"cross_iteration_corr":d.get("cross_iteration_corr")}
    elif label == "G":
        out[label] = {"n_gpus_available":d.get("n_gpus_available"),"profile":d.get("profile"),"scaling_estimate":d.get("scaling_estimate")}
    elif label == "H":
        out[label] = {k:d[k] for k in d if k not in ("schema_version","phase")}
summary_path = os.path.join(root,"c30_extracted_summary.json")
json.dump(out, open(summary_path,"w"), indent=2)
print(f"Saved summary ({sum(len(json.dumps(v)) for v in out.values())} bytes)")

#!/usr/bin/env python3
"""Report puller: per-K cell metrics + per-group breakdown for the four cells."""
import json
import os

A = "/root/epic05_mg0/artifacts"
for s in ("room", "bicycle"):
    v = json.load(open(os.path.join(A, f"MG0_VARIANCE_{s.upper()}.json")))
    for st in ("20000", "25000"):
        m = v["checkpoints"][st]
        print("== %s@%s  G_norm=%.4g  loss_mean=%.5f  views=%d" %
              (s, st, m["G_norm"], m["loss_mean"], m["n_views"]))
        for K in ("2048", "8192", "32768"):
            r = m["per_K"][K]
            print("  K=%-5s raw_ratio=%.4f adam_ratio=%.4g med_gain=%+.4f "
                  "p10_gain=%+.4f norm_ratio=%.3f pos=%.3f" %
                  (K, r["raw_var_ratio"], r["adam_var_ratio"],
                   r["median_cosine_gain"], r["p10_cosine_gain"],
                   r["median_norm_ratio_to_G"], r["positive_dot_fraction"]))
        r = m["per_K"]["32768"]
        print("  per-group @K=32768:")
        for g, rec in r["per_group"].items():
            print("    %-9s raw_ratio=%.4f adam_ratio=%.4g med_gain=%+.4f "
                  "raw_var=%.4g hotcv_var=%.4g" %
                  (g, rec["raw_var_ratio"], rec["adam_var_ratio"],
                   rec["median_cosine_gain"], rec["raw_var"], rec["hotcv_var"]))
print()
print("== control maintenance (max over cells) ==")
c = json.load(open(os.path.join(A, "MG0_COST.json")))
for K in ("2048", "8192", "32768"):
    rows = [(k, v["event_ms"]["median"], v["control_maintenance_ratio"],
             v["host_wall_ms_per_repeat"], v["control_maintenance_ratio_host_wall"])
            for k, v in c["cells"].items() if k.endswith(K)]
    ev = [r[1] for r in rows]
    rt = [r[2] for r in rows]
    print("K=%-5s event_ms med range [%.4f, %.4f]  ratio range [%.5f, %.5f]" %
          (K, min(ev), max(ev), min(rt), max(rt)))

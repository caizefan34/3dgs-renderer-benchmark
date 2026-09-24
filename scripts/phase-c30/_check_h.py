import json
d=json.load(open("results/phase-c30/c30_h_precision.json"))
fp32=d["fp32"]
print(f"FP32: fwd={fp32['mean_fwd_ms']:.2f} bwd={fp32['mean_bwd_ms']:.2f} opt={fp32['mean_opt_ms']:.2f} t_iter={fp32['mean_t_iter_ms']:.2f}")
print(f"TF32: fwd={d['tf32']['mean_fwd_ms']:.2f} bwd={d['tf32']['mean_bwd_ms']:.2f} t_iter={d['tf32']['mean_t_iter_ms']:.2f}")
print(f"MP final Gs: {d['mp']['final_gaussians']:,}")

d2=json.load(open("results/phase-c30/c30_e_loss_frequency.json"))
t=d2["component_timing_ms"]
print(f"L1={t['mean_l1_ms']:.4f}ms DSSIM={t['mean_dssim_ms']:.4f}ms bwd={t['mean_bwd_ms']:.2f}ms")

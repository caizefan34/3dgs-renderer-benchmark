import json
d=json.load(open("/home/liaoyuanjun/cgt_g5_native/results/G5A_NATIVE_CAPACITY.json"))
print("G5-A verdict:",d["verdict"],"gate_pass:",d["native_variants_gate_pass_all_scenes"])
print("SSIM self-test (GT vs GT, must be 1.0):",d["sanity_frozen_chain"]["ssim_selftest_gt_vs_gt"])
print("frozen chain max_abs (must be 0.0):",d["sanity_frozen_chain"]["frozen_student_vs_ref_student_max_abs"])
print("teacher-vs-student median PSNR:",round(d["sanity_frozen_chain"]["teacher_vs_student_median_psnr"],2))
for s,r in d["per_scene"].items():
    A=r["variants"]["A_frozen_teacher"]; B=r["variants"]["B_native_from_teacher"]; C=r["variants"]["C_native_from_scratch"]
    st=r["gap_stratified_by_brightness"]
    print(f"[{s}] n={r['n_tiles']} A_psnr_med={A['psnr_median']:.2f} A_ssim={A['ssim_mean']:.4f} | B dPSNR={B['dPSNR_vs_A_median']:+.2f} dSSIM={B['dSSIM_vs_A_mean']:+.4f} | C dPSNR={C['dPSNR_vs_A_median']:+.2f} dSSIM={C['dSSIM_vs_A_mean']:+.4f}")
    print(f"      strat dark={st['dark']['mean_gapB_psnr']:.2f} mid={st['mid']['mean_gapB_psnr']:.2f} bright={st['bright']['mean_gapB_psnr']:.2f} | per_tile_rows={len(r['per_tile'])}")
b=json.load(open("/home/liaoyuanjun/cgt_g5_native/results/G5B_GRADIENT_CHECK.json"))
print("G5-B verdict:",b["verdict"],"cosine_min:",b["measured"]["cosine_min"],"rel_l2_max:",b["measured"]["rel_l2_max"],"max_abs:",b["measured"]["max_abs_max"],"nan:",b["measured"]["nan_inf_total"],"n_cases:",b["n_cases"])

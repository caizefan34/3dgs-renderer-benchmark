import json

r = json.load(open("/mnt/storage_pool/liaoyuanjun/final30k_gates_compat/smoke_2k_results.json"))
print("VERDICT:", r["verdict"])
print("missing:", r["missing"])
for s, p in r["per_scene"].items():
    sm = p["summary"]
    print(f"\n=== {s}: {p['verdict']}")
    if p["fail_reasons"]:
        print("  FAIL:", p["fail_reasons"])
    print(f"  N: {sm['initial_N']:,} -> b1a {sm['final_N']['b1a']:,} | c0 {sm['final_N']['c0']:,}")
    print(f"  clones b1a {sm['total_clones']['b1a']:,} c0 {sm['total_clones']['c0']:,} | "
          f"splits b1a {sm['total_splits']['b1a']:,} c0 {sm['total_splits']['c0']:,}")
    print(f"  PSNR b1a {sm['final_psnr']['b1a']:.3f} c0 {sm['final_psnr']['c0']:.3f} "
          f"(delta {sm['final_psnr']['c0'] - sm['final_psnr']['b1a']:+.3f} dB)")
    print(f"  SSIM b1a {sm['final_ssim']['b1a']:.4f} c0 {sm['final_ssim']['c0']:.4f}")
    print(f"  loss_last b1a {sm['loss_last']['b1a']:.5f} c0 {sm['loss_last']['c0']:.5f}")
    c = p["gate_checks"]
    print(f"  tail ratios (last 5 matched): {c['trajectory']['tail_ratios']}")
    print(f"  max N ratio over run: {c['trajectory']['max_ratio']}")
    print(f"  clone/split/finalN ratios: {c['total_clones_ratio']} / {c['total_splits_ratio']} / {c['final_N_ratio']}")
    print(f"  growth factor b1a {c['growth_factor']['b1a']:.3f} c0 {c['growth_factor']['c0']:.3f}")
    print(f"  grad_inf events b1a {c['finite']['events_with_grad_inf']['b1a']} "
          f"c0 {c['finite']['events_with_grad_inf']['c0']}")

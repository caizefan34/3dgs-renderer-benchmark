import json, os
os.chdir("/root/epic05_cg0/out")
cells=["room@15000","room@25000","bicycle@15000","bicycle@25000"]
for c in cells:
    q=json.load(open(c+"/CG0_QUALITY.json"))
    s=json.load(open(c+"/CG0_SUMMARY.json"))
    st=json.load(open(c+"/CG0_COMPILER_STATS.json"))
    t=json.load(open(c+"/CG0_TIMING.json"))
    print("===",c,"===")
    print("  N_source=%d N_compiled=%d count_ratio=%.4f h=%.4f fallback=%.3f"%(
        st["N_source"],st["N_compiled"],s["count_ratio"],st["voxel_width_h"],st["sh_fallback_frac"]))
    print("  src trainval psnr=%.2f holdout psnr=%.2f"%(q["source"]["trainval16"]["psnr"],q["source"]["holdout16"]["psnr"]))
    print("  compiled0 holdout delta=%.2f trainval delta=%.2f"%(q["compiled0"]["holdout16"]["psnr"],q["compiled0"]["trainval16"]["psnr"]))
    print("  compiled500 holdout delta=%.2f"%q["compiled500"]["holdout16"]["psnr"])
    print("  compiled1500 holdout delta=%.2f worst=%.2f trainval delta=%.2f ssim=%.4f"%(
        q["compiled1500"]["holdout16"]["psnr"],q["compiled1500"]["holdout16"]["ssim"],
        q["compiled1500"]["trainval16"]["psnr"],q["compiled1500"]["trainval16"]["ssim"]))
    print("  cond_frac=%.4f oracle_incl=%.4f refine1500=%.1fs"%(
        t["conditional_compile_refine_fraction"],t["oracle_inclusive_fraction"],t["refine_1500_s"]))

import pandas as pd, numpy as np
pd.set_option("display.width",200)
for scene in ["room","bicycle"]:
    df=pd.read_parquet(f"/root/glm0/work/uidtable_{scene}.parquet")
    print(f"\n========== {scene}  N={len(df)} ==========")
    print("birth_type:", df.birth_type.value_counts().to_dict())
    print("terminal_reason:", df.terminal_reason.value_counts().to_dict())
    print("age (alive->30000): mean=%.0f median=%.0f p90=%.0f"%(df.age.mean(),df.age.median(),df.age.quantile(.9)))
    cols=["vis_frac","vis_count","tiles_cum","cam_count","grad_mean_norm","last_absgrad_norm_px","parent_opacity","scale_volume","threshold_margin","footprint_tiles_last_obs"]
    print(df[cols].describe(percentiles=[.1,.25,.5,.75,.9,.99]).T.to_string())
    # visibility vs contribution correlation
    sub=df[df.vis_count>0]
    print("corr(vis_frac, log1p tiles_cum)=%.3f"%np.corrcoef(sub.vis_frac, np.log1p(sub.tiles_cum))[0,1])
    print("corr(log vis_count, log tiles_cum)=%.3f"%np.corrcoef(np.log1p(sub.vis_count), np.log1p(sub.tiles_cum))[0,1])
    # fraction never visible in sampled window
    print("frac vis_count==0: %.3f"%(df.vis_count==0).mean())
    # visibility fraction distribution
    print("vis_frac deciles:", np.round(df.vis_frac.quantile(np.arange(0,1.01,.1)).values,3))

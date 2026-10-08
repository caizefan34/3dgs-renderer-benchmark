import pandas as pd, numpy as np, json
ART="/root/epic05_d0/artifacts"
# Population trajectory
pt=pd.read_parquet(f"{ART}/D0_POPULATION_TRAJECTORY.parquet")
print("=== POPULATION_TRAJECTORY ===")
print("cols:", pt.columns.tolist())
print("shape:", pt.shape)
print(pt.head(3).to_string())
print("...")
print(pt.tail(2).to_string())
# Cost samples
cs=pd.read_parquet(f"{ART}/D0_RENDER_COST_SAMPLES.parquet")
print("\n=== COST_SAMPLES ===")
print("cols:", cs.columns.tolist())
print("shape:", cs.shape)
print("scene_id vals:", cs["scene_id"].unique().tolist() if "scene_id" in cs else "n/a")
print("n distinct iters:", cs["iter"].nunique() if "iter" in cs else "n/a")
print("iter range:", (cs["iter"].min(), cs["iter"].max()) if "iter" in cs else "n/a")
print("n distinct uids:", cs["uid"].nunique() if "uid" in cs else "n/a")
print(cs.head(5).to_string())

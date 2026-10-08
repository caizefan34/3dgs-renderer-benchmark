import pandas as pd, numpy as np
lin=pd.read_parquet("/root/epic05_d0/artifacts/D0_LINEAGE_OUTCOMES_room.parquet")
print("lin cols:", lin.columns.tolist())
print("lin shape:", lin.shape)
print(lin.head(3).to_string())
# cost sample per-iter visible count vs population n_visible
cs_iters=pd.read_parquet("/root/epic05_d0/artifacts/D0_RENDER_COST_SAMPLES.parquet", columns=["scene_id","iter"])
print("scene_id iters:", cs_iters.groupby("scene_id")["iter"].agg(["min","max","nunique"]).to_string())

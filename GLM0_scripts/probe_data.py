import pandas as pd
print("===duckdb===")
try:
    import duckdb; print("duckdb", duckdb.__version__)
except Exception as e:
    print("no duckdb:", e)
print("===polars===")
try:
    import polars; print("polars", polars.__version__)
except Exception as e:
    print("no polars:", e)
print("===birth table room===")
bt=pd.read_parquet("/root/epic05_d0/artifacts/D0_BIRTH_TABLE_room.parquet")
print("cols:", bt.columns.tolist())
print("shape:", bt.shape)
cols=[c for c in ["uid","parent_uid","birth_iter","birth_type","root_uid","vis_count_128","vis_frac_128","grad_mean_norm","footprint_tiles_last_obs","parent_opacity","threshold_margin","last_absgrad_norm_px"] if c in bt.columns]
print(bt[cols].head(5).to_string())

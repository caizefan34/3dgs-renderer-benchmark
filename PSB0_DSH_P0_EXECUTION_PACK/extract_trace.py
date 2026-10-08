import json
t=json.load(open("/root/dsh_20261001/artifacts/DSH7_BARS0/BARS0_TILE_TRACE.json"))
scenes=t["scenes"]
out={"provenance":{"source":"DSH7_BARS0/BARS0_TILE_TRACE.json","K":t["K"],"scenes":list(scenes.keys()),
     "note":"Real FasterGSFused fused-backward tile traces, nerficg harness, 1920x1080, seed 42"}}
def agg(lst):
    import statistics as st
    keys=["p10","p50","p90","p99","mean","max","min","count"]
    return {k:(round(st.mean([d[k] for d in lst]),4) if k in("mean",) else None) for k in []} or None
for name,sc in scenes.items():
    rec={}
    # last iteration = final_record
    final=sc.get("splat_list_len")[-1]
    rec["final_iteration_tag"]=None
    # collect per-metric mean over iterations for stable distributions
    metrics=["splat_list_len","max_contrib","n_k32_buckets","partial_bucket_occupancy","forward_checkpoint_bytes","backward_bucket_count","mean_replay_len"]
    rec["metrics_mean_over_iters"]={}
    for m in metrics:
        if m in sc:
            arr=sc[m]
            rec["metrics_mean_over_iters"][m]={k:round(sum(d[k] for d in arr)/len(arr),4) for k in ["p10","p50","p90","p99","mean","max","min"] if k in arr[0]}
    rec["final_record_selected"]={k:sc[k][-1] for k in ["splat_list_len","max_contrib","n_k32_buckets","mean_replay_len","forward_checkpoint_bytes"] if k in sc}
    out[name]=rec
print(json.dumps(out,indent=1))

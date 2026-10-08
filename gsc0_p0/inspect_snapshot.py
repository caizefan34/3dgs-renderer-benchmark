import torch, json, os, glob
d = "/root/gsc0_smoke"
idx = json.load(open(os.path.join(d, "trace_index_room.json")))
print("=== trace_index_room.json ===")
for k in ["scene","iterations","initial_N","final_N","total_clones","total_splits","total_prunes","n_cameras","scene_extent","phases","elapsed_s"]:
    print(f"  {k}: {idx.get(k)}")
for r in idx["trace_records"]:
    print("  rec:", {k:r[k] for k in ["phase","iteration","N","active_sh_degree","bytes"]})
f = sorted(glob.glob(os.path.join(d,"traces","*.pt")))[0]
print("\n=== snapshot", os.path.basename(f), "===")
s = torch.load(f, map_location="cpu")
for k in ["scene","phase","iteration","N","active_sh_degree","spatial_lr_scale"]:
    print(f"  {k}: {s[k]}")
for k in ["ids","xyz","scaling","rotation","opacity","shs","grad_sum","vis_count","xyz_gradient_accum","denom","max_radii2D"]:
    t = s[k]
    nan = torch.isnan(t.float()).sum().item() if t.is_floating_point() else "n/a"
    print(f"  {k}: shape={tuple(t.shape)} dtype={t.dtype} nan={nan} min={t.float().min().item():.4g} max={t.float().max().item():.4g}")
print("  adam keys:", list(s["adam"].keys()))
for name, st in s["adam"].items():
    ea, eas = st["exp_avg"], st["exp_avg_sq"]
    print(f"    adam[{name}]: exp_avg {tuple(ea.shape)} nan={torch.isnan(ea).sum().item()} | exp_avg_sq {tuple(eas.shape)} nan={torch.isnan(eas).sum().item()}")
# consistency: ids unique, grad gathered by id
assert s["ids"].dim()==1 and s["ids"].shape[0]==s["N"], "ids shape mismatch"
assert s["ids"].unique().numel()==s["N"], "ids not unique"
assert s["shs"].shape[0]==s["N"] and s["shs"].shape[1]==s["active_sh_degree"], "shs shape mismatch"
assert s["grad_sum"].shape[0]==s["N"] and s["vis_count"].shape[0]==s["N"], "gather shape mismatch"
print("\n  CONSISTENCY OK: ids unique, shapes match N, shs coeffs == active_sh_degree")
print(">>> INSPECT SUCCESS")
import torch, numpy as np, glob, importlib.util
spec = importlib.util.spec_from_file_location("gsc0a", "/root/gsc0_analysis.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dc = torch.load(sorted(glob.glob("/root/gsc0_run/room/traces/*densify_complete*.pt"))[-1], map_location="cpu")
mat = torch.load(sorted(glob.glob("/root/gsc0_run/room/traces/*mature*.pt"))[-1], map_location="cpu")
common, ia, ib = m.match_by_id(dc, mat)
print("matched", common.size, "of dc", dc["N"], "mat", mat["N"])
# pick 5 random matched Gaussians and show raw + rel change
gidx = np.random.RandomState(0).choice(common.size, 5, replace=False)
EPS = m.EPS
for k in gidx:
    i, j = ia[k], ib[k]
    xyz_d = dc["xyz"][i]; xyz_m = mat["xyz"][j]
    sc_d = dc["scaling"][i]; sc_m = mat["scaling"][j]
    op_d = dc["opacity"][i]; op_m = mat["opacity"][j]
    r_xyz = (xyz_m-xyz_d).norm().item()/max(xyz_m.norm().item(),EPS)
    r_sc = (sc_m-sc_d).norm().item()/max(sc_m.norm().item(),EPS)
    r_op = abs(op_m-op_d).item()/max(abs(op_m.item()),EPS)
    print(f"  id={int(common[k])} xyz|d|={xyz_d.norm():.3f} xyz|m|={xyz_m.norm():.3f} r_xyz={r_xyz:.3e} | logscale_d={sc_d.tolist()} r_sc={r_sc:.3e} | op_d={op_d.item():.3f} op_m={op_m.item():.3f} r_op={r_op:.3e}")
print(">>> manual spot-check OK")
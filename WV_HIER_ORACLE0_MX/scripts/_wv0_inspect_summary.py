import torch
s=torch.load("/dev/shm/wv0_out/WV0_GRADIENT_SUMMARY.pt", weights_only=False)
grad=s["grad"]
sid="room__iter5000"
c=grad[sid]["G8_p0.4"]
print("geom_nmse_mean:", c.get("geom_nmse_mean"))
print("geom_cos_mean:", c.get("geom_cos_mean"))
print("macro_nmse_mean:", c.get("macro_nmse_mean"))
for g in ["means","quats","scales","opacities","colors"]:
    gv=c.get(g,{})
    print(f"  {g}: nmse={gv.get('nmse')} cos={gv.get('cos')}")

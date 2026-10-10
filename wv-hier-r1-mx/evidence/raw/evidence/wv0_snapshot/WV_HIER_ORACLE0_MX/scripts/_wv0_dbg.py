import torch, numpy as np
ALL_GROUPS=["means","quats","scales","opacities","colors"]
GEOM_GROUPS=["means","quats","scales"]
def macro(vals):
    vals=[v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None
s=torch.load("/dev/shm/wv0_out/states/room__iter5000.pt", weights_only=False)
c=s["conditions"]["G8_p0.4"]
print("c keys:", list(c.keys()))
print("c[groups] keys:", list(c["groups"].keys()))
gv=c["groups"]["means"]
print("means keys:", list(gv.keys()))
print("means nmse:", gv["nmse"])
print("direct mean:", c["groups"]["means"]["nmse"]["mean"])
print("list:", [c["groups"][g]["nmse"]["mean"] for g in GEOM_GROUPS])
print("macro:", macro([c["groups"][g]["nmse"]["mean"] for g in GEOM_GROUPS]))

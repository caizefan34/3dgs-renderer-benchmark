import torch, numpy as np
ALL_GROUPS=["means","quats","scales","opacities","colors"]
GEOM_GROUPS=["means","quats","scales"]
def macro(vals):
    vals=[v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None
s=torch.load("/dev/shm/wv0_out/states/room__iter5000.pt", weights_only=False)
c=s["conditions"]["G8_p0.4"]
# replicate exactly
for metric in ("nmse","cos","rel_l2"):
    vals=[c["groups"][g][metric]["mean"] for g in GEOM_GROUPS]
    print(metric, "vals=", vals, "macro=", macro(vals))

import sys
sys.path.insert(0, "/root/epic05_n1")
import torch
import n1_core as C
C.bootstrap_c0()
ckpt = C.load_checkpoint("room", 25000)
model = C.build_model(ckpt)
C.scene_reset()
o = model.get_opacity.detach()
eps = 1e-7
o_c = torch.clamp(o, 0.0, 1.0 - eps)
gamma = 1.0
A = 1.0 - (1.0 - o_c) ** gamma
B = -torch.expm1(gamma * torch.log1p(-o_c))
C_form = torch.where(torch.tensor(gamma) == 1.0, o_c, 1.0 - (1.0 - o_c) ** gamma)
for name, X in [("A literal", A), ("B stable", B), ("C where-endpoint", C_form)]:
    d = (X - o).abs()
    print(f"{name:18s}: max|d|={float(d.max()):.3e} mean|d|={float(d.mean()):.3e} num_changed={int((X!=o).sum())} num>1e-9={int((d>1e-9).sum())}")
g2 = 0.84
A2 = 1.0 - (1.0 - o_c) ** g2
B2 = -torch.expm1(g2 * torch.log1p(-o_c))
print(f"gamma=0.84 A vs B: max|d|={float((A2-B2).abs().max()):.3e} (same formula, ~0 expected)")

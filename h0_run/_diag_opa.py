import sys
sys.path.insert(0, "/root/epic05_n1")
import torch
import n1_core as C

C.bootstrap_c0()
ckpt = C.load_checkpoint("room", 25000)
model = C.build_model(ckpt)
C.scene_reset()
o = model.get_opacity.detach()  # sigmoid(_opacity).squeeze(-1), 1D
eps = 1e-7
o_c = torch.clamp(o, 0.0, 1.0 - eps)
o_g = 1.0 - (1.0 - o_c) ** 1.0
diff = (o_g - o).abs()
print("N gaussians:", o.numel())
print("opacity min/max/mean:", float(o.min()), float(o.max()), float(o.mean()))
print("clamp-active count (o > 1-eps):", int((o > 1.0 - eps).sum()))
print("o>0.999 count:", int((o > 0.999).sum()))
print("o>0.99 count:", int((o > 0.99).sum()))
print("max |o_g - o|:", float(diff.max()))
print("mean |o_g - o|:", float(diff.mean()))
print("num changed (o_g != o):", int((o_g != o).sum()))
print("num changed by >1e-9:", int((diff > 1e-9).sum()))
# what if we clamp at a tighter domain (eps=0 -> no upper clamp)?
o_c0 = torch.clamp(o, 0.0, 1.0)
o_g0 = 1.0 - (1.0 - o_c0) ** 1.0
print("with eps=0: max |o_g0 - o|:", float((o_g0 - o).abs().max()), "num changed:", int((o_g0 != o).sum()))

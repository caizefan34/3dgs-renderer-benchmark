import torch

b0 = torch.load("/mnt/storage_pool/liaoyuanjun/pubphase/b0gate/b0.pt")
b1 = torch.load("/mnt/storage_pool/liaoyuanjun/pubphase/b0gate/b1.pt")

g0 = b0["gxyz"].float()
g1 = b1["gxyz"].float()
n0 = g0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
n1 = g1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
cos = (g0 * g1).sum(-1) / (n0.squeeze(-1) * n1.squeeze(-1))

both = (n0.squeeze(-1) > 1e-9) & (n1.squeeze(-1) > 1e-9)
print(f"N={g0.shape[0]} both-nonzero={int(both.sum())}")
print(f"mean cos: {float(cos[both].mean()):.4f}")
print(f"median cos: {float(cos[both].median()):.4f}")
print(f"frac cos>0: {float((cos[both] > 0).float().mean()):.4f}")
print(f"frac cos>0.5: {float((cos[both] > 0.5).float().mean()):.4f}")
# norm-ratio distribution (scale agreement)
ratio = (n0.squeeze(-1) / n1.squeeze(-1))[both]
print(f"norm ratio p10/p50/p90: {float(ratio.quantile(0.1)):.3f} / {float(ratio.quantile(0.5)):.3f} / {float(ratio.quantile(0.9)):.3f}")

# frame stats
f0, f1 = b0["frame"], b1["frame"]
mse = float(((f0 - f1) ** 2).mean())
import math
print(f"frame PSNR: {10*math.log10(1/max(mse,1e-12)):.2f} dB")
print(f"frame max abs diff: {float((f0-f1).abs().max()):.4f}  mean abs diff: {float((f0-f1).abs().mean()):.5f}")

# visibility
r0, r1 = b0["radii_pos"], b1["radii_pos"]
inter = int((r0 & r1).sum()); union = int((r0 | r1).sum())
print(f"vis: b0={int(r0.sum())} b1={int(r1.sum())} jaccard={inter/union:.4f}")

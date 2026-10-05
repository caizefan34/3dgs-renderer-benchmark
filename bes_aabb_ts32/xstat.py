import torch
d="out/ts"
b16=torch.load(d+"/img_ts16_cam1.pt", map_location="cpu")["base"]
b32=torch.load(d+"/img_ts32_cam1.pt", map_location="cpu")["base"]
diff=(b16-b32).abs()
print("num pixels", b16.shape[0])
print("max diff", diff.max().item())
print("mean diff", diff.mean().item())
for thr in [1e-4,1e-3,1e-2,0.1,0.5]:
    print("frac > %.0e: %.4f"%(thr, (diff>thr).float().mean().item()))
idx=diff.argmax()
print("max-diff pixel idx", idx.item(), "b16", b16[idx].tolist(), "b32", b32[idx].tolist())

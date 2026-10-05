import torch
d="out/ts"
b16=torch.load(d+"/img_ts16_cam1.pt", map_location="cpu")
b32=torch.load(d+"/img_ts32_cam1.pt", map_location="cpu")
def md(a,b): return (a-b).abs().max().item()
print("shapes", tuple(b16["base"].shape), tuple(b32["base"].shape))
print("cross-TS base16 vs base32 :", md(b16["base"], b32["base"]))
print("cross-TS tile16 vs tile32 :", md(b16["tile"], b32["tile"]))
print("within TS16 base vs tile  :", md(b16["base"], b16["tile"]))
print("within TS32 base vs tile  :", md(b32["base"], b32["tile"]))
print("cross base16 vs tile32    :", md(b16["base"], b32["tile"]))

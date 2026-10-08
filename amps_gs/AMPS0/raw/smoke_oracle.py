import sys, torch, math
sys.path.insert(0, ".")
from momentgrad_gs.scene import load_scene
from momentgrad_gs.oracle import project_gaussians, rasterize
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
cams, init = load_scene("data/datasets/mipnerf360/room", split="train", every=8, max_points=400, device=device)
print("loaded", len(cams), "cams, N=", init.means.shape, "device", device)
cam = cams[0]
H, W = cam.height//8, cam.width//8
init.means.requires_grad_(True)
proj2 = project_gaussians(init.means, init.quats, init.scales, torch.sigmoid(init.opacities),
    cam.worldtocam, cam.K, colors_dc=init.colors, max_radius=256)
res2 = rasterize(proj2, H, W, pixel_loss_grad=torch.ones_like(res2.image) if False else torch.ones(H,W,3,device=device), tile=16)
loss = res2.image.sum()
loss.backward()
print("image", tuple(res2.image.shape), "n_interactions", res2.n_interactions)
print("dL/dmeans finite:", bool(torch.isfinite(init.means.grad).all()), "norm:", float(init.means.grad.norm()))
print("SMOKE_OK")

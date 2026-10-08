import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
dev="cuda"
N=50
means2D=torch.zeros(N,2,device=dev,requires_grad=True)
means=torch.randn(N,3,device=dev)
colors=torch.rand(N,3,device=dev,requires_grad=True)
opacities=torch.rand(N,1,device=dev)
scales=torch.rand(N,3,device=dev)
rotations=torch.nn.functional.normalize(torch.randn(N,4,device=dev),dim=-1)
cam=torch.zeros(3,device=dev)
H=W=32
ws=torch.zeros(3,device=dev)
settings=GaussianRasterizationSettings(
    image_height=H, image_width=W, tanfovx=0.5, tanfovy=0.5, bg=ws,
    scale_modifier=1.0, viewmatrix=torch.eye(4,device=dev),
    projmatrix=torch.eye(4,device=dev), sh_degree=0, campos=torch.zeros(3,device=dev),
    prefiltered=False, debug=False)
rasterizer=GaussianRasterizer(raster_settings=settings)
rendered, radii = rasterizer(
    means=means, means2D=means2D, shs=None, colors_precomp=colors,
    opacities=opacities, scales=scales, rotations=rotations, cov3D_precomp=None)
print("Inria render OK; rendered", rendered.shape, "radii", radii.shape)
loss=rendered.sum(); loss.backward()
print("means2D.grad is None?", means2D.grad is None, "colors.grad", None if colors.grad is None else colors.grad.shape)

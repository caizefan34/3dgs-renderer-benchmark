import numpy as np
from PIL import Image
def load(p): return np.asarray(Image.open(p),dtype=np.float64)/255.0
gt=load("/root/epic05_cg0/png_gt.png")
for tag in ["png_src","png_cmp_1.0","png_cmp_0.3"]:
    a=load(f"/root/epic05_cg0/{tag}.png")
    mse=np.mean((a-gt)**2)
    psnr=99.0 if mse<1e-10 else 10*np.log10(1.0/mse)
    print(f"{tag} psnr={psnr:.2f} mse={mse:.5f} img_mean={a.mean():.3f}")

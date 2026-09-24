import lpips
import torch

m = lpips.LPIPS(net="vgg").cuda()
x = torch.rand(1, 3, 64, 64).cuda() * 2 - 1
y = torch.rand(1, 3, 64, 64).cuda() * 2 - 1
print("LPIPS_OK self=", float(m(x, x)), "rand=", float(m(x, y)))

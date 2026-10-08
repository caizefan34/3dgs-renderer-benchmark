try:
    print(open("/root/epic05_py_wrapper.sh").read())
except Exception as e:
    print("no wrapper:", e)
import fused_ssim
print("fused_ssim:", fused_ssim.__file__)
print("fused_ssim attrs:", [a for a in dir(fused_ssim) if not a.startswith("_")])
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())

import inspect
import fused_ssim
print("FILE", inspect.getsourcefile(fused_ssim))
print("DIR", [x for x in dir(fused_ssim) if not x.startswith("__")])
src = inspect.getsource(fused_ssim.fused_ssim)
print("=== fused_ssim source ===")
print(src)

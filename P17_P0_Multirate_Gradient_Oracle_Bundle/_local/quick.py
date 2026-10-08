import sys, subprocess
print("=== bicycle ckpt ===")
subprocess.run([sys.executable, "/root/epic05_p17/inspect_ckpt.py",
                "/root/epic05_p13x/runs/bicycle_diag/ckpt_16000.pt"])
print("=== wrapper ===")
try:
    print(open("/root/epic05_py_wrapper.sh").read())
except Exception as e:
    print("no wrapper:", e)
import fused_ssim
print("fused_ssim:", fused_ssim.__file__)
print("fused_ssim attrs:", [a for a in dir(fused_ssim) if not a.startswith("_")])

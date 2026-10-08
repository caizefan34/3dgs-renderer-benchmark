import fused_ssim_cuda as F
print([x for x in dir(F) if not x.startswith("__")])
for name in ["fusedssim","fusedssim_backward"]:
    fn = getattr(F, name, None)
    if fn is not None:
        print("===", name, "===")
        print(getattr(fn,"__doc__",None))

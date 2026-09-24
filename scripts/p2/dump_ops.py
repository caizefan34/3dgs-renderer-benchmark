import importlib.util, torch
CORE="/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so"
spec=importlib.util.spec_from_file_location("gsplat_cuda", CORE)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
try:
    all_ops = sorted(dir(torch.ops.gsplat))
    print("TOTAL", len(all_ops))
    for o in all_ops:
        print(o)
except Exception as e:
    print("ERR", repr(e))
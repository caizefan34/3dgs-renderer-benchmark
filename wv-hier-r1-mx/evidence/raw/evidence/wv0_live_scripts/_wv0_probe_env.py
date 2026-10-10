import sys
try:
    import torch, gsplat
    print("PYOK", sys.version.split()[0], "torch", torch.__version__, "gsplat", gsplat.__version__)
    print("cuda_avail", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("dev", torch.cuda.get_device_name(0))
except Exception as e:
    print("ERR", repr(e)[:300])

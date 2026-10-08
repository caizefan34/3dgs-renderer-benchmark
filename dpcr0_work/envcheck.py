import sys
sys.path.insert(0, "/root/epic05_n1")
import n1_core as C
print("n1 OK; C0_SO=", C.C0_SO)
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "dev", torch.cuda.get_device_name(0))
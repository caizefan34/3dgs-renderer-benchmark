import torch, sys
ckpt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print(sorted(ckpt.keys()))

import sys, os
P = "/mnt/storage_pool/liaoyuanjun/fused_profile0"
sys.path.insert(0, P+"/FUSED_HOTSPOT0/scripts")
for _p in [P+"/pylibs", P+"/methods_overlay", "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/src", "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/scripts"]:
    if _p not in sys.path: sys.path.insert(0, _p)
import torch
from hotspot import build, make_iter
ck = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_room_2026-10-06-01-01-33/checkpoints/final.pt"
trainer, dataset, train_ds, sampler, ad, g = build(P+"/profile/mat_room.yaml", ck, 2)
_iter = make_iter(trainer.renderer, trainer.loss)
for i in range(8):
    _iter(train_ds, sampler, ad, g, 100000+i)
torch.cuda.synchronize()
print("NCU_ITER_DONE")
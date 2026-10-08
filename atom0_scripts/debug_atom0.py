import sys, os, torch
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/fused_profile0/ATOM0/scripts")
import atom0_common as A
CFG="/mnt/storage_pool/liaoyuanjun/fused_profile0/profile/mat_room.yaml"
CK="/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_room_2026-10-06-01-01-33/checkpoints/final.pt"
trainer, dataset, train_ds, sampler, ad, g = A.build_model(CFG, CK, 0)
n_prim = int(g.means.shape[0])
view = train_ds[0]
settings, bg = A.view_settings(trainer, view, g)
print("bg", bg.shape, bg.dtype, bg.device, flush=True)
fd = A.call_forward(g, settings, bg)
print("FWD ok ninst", fd['n_instances'], "nbuckets", fd['n_buckets'], flush=True)
print("blob bytes pb/tb/ib/bb", fd['pb'].numel(), fd['tb'].numel(), fd['ib'].numel(), fd['bb'].numel(), flush=True)
print("active_sh_bases", g.active_sh_bases, flush=True)
inst = A.get_instance_prim(fd)
print("inst_prim ok", inst.shape, inst.dtype, inst[:5].tolist(), flush=True)
grad_image = torch.randn_like(fd['image'])
gm=torch.zeros(n_prim,2,device='cuda'); gc=torch.zeros(3*n_prim,device='cuda')
go=torch.zeros(n_prim,1,device='cuda'); gcol=torch.zeros(n_prim,3,device='cuda')
EMPTY=torch.empty(0,device='cuda')
print("calling backward_atom0 variant0 full...", flush=True)
ms,_ = A.CATOM.backward_atom0(grad_image, fd['image'], bg, fd['pb'],fd['tb'],fd['ib'],fd['bb'],
    n_prim, settings.width, settings.height, fd['n_instances'], fd['n_buckets'], fd['sel'],
    0, 0, fd['n_buckets'], gm,gc,go,gcol, EMPTY, EMPTY, 0)
torch.cuda.synchronize()
print("variant0 full ms", ms, "gm_norm", gm.norm().item(), flush=True)

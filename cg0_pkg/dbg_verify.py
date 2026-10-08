import sys
sys.path.insert(0, "/root/epic05_n1")
sys.path.insert(0, "/root/epic05_frozen/higs_c0_final30k_worktree")
sys.path.insert(0, "/root/epic05_build_cache/scene/gsplat_scene_cuda")
sys.path.insert(0, "/root/3dgs-renderer-benchmark/scripts/epic05/phase7")
sys.path.insert(0, "/root/3dgs-renderer-benchmark/baseline/reference_v1")
import torch
import n1_core as C
import cg0_prereg as P
import cg0_compiler as COM
import cg0_appearance as APP
C.bootstrap_c0()
scene="room"; it=15000
ds=C.load_scene_dataset(scene)
ck=C.load_checkpoint(scene,it)
cam=ds.get_camera(0)
xyz=ck["xyz"].cuda()
K=cam.K; fx,fy,cx,cy=K[0,0].item(),K[1,1].item(),K[0,2].item(),K[1,2].item()
for k in [0,100000,400000]:
    X=xyz[k:k+1]
    u,v,z,inf=APP._project(cam,X)
    uc=u.round().long().clamp(0,1919); vc=v.round().long().clamp(0,1079)
    d=z
    Xb=COM.backproject(cam, uc.float(), vc.float(), d)
    u2,v2,z2,inf2=APP._project(cam, Xb)
    print(f"k={k} orig(u,v,z)=({u.item():.1f},{v.item():.1f},{z.item():.3f}) backproj_reproj=({u2.item():.1f},{v2.item():.1f},{z2.item():.3f}) err={(Xb-X).norm().item():.5f}")

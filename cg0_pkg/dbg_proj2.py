import sys
sys.path.insert(0, "/root/epic05_n1"); sys.path.insert(0, "/root/epic05_frozen/higs_c0_final30k_worktree"); sys.path.insert(0, "/root/epic05_build_cache/scene/gsplat_scene_cuda"); sys.path.insert(0, "/root/3dgs-renderer-benchmark/scripts/epic05/phase7"); sys.path.insert(0, "/root/3dgs-renderer-benchmark/baseline/reference_v1")
import torch, numpy as np
import n1_core as C
import cg0_prereg as P
import cg0_compiler as COM
import cg0_appearance as APP
C.bootstrap_c0()
scene="room"
ds=C.load_scene_dataset(scene)
cam=ds.get_camera(0)
R=cam.viewmatrix[:3,:3]; t=cam.viewmatrix[:3,3]
Ccenter=cam.camera_center
Ccheck=-(R.t()@t)
print("camera_center prop=",Ccenter.detach().cpu().numpy()," -R^T t=",Ccheck.detach().cpu().numpy()," match=",float((Ccenter-Ccheck).norm()))
# synthetic point at cam-frame (0,0,-5)
cam_pt=torch.tensor([[0.0,0.0,-5.0]],device="cuda")
world=cam_pt@R + Ccenter.unsqueeze(0)   # cam_pt @ R + C  (row-vector form)
print("world=",world.detach().cpu().numpy())
u,v,z,inf=APP._project(cam,world)
print("_project u=%.3f v=%.3f z=%.5f inf=%s (expect u~960 v~540 z~5)"%(float(u),float(v),float(z),bool(inf)))
Xb=COM.backproject(cam,u,v,z)
print("backproject roundtrip err=%.6f"%float((Xb-world).norm()))
# now test a non-center point cam_pt=(0.1,-0.05,-5)
cam_pt2=torch.tensor([[0.1,-0.05,-5.0]],device="cuda")
w2=cam_pt2@R + Ccenter.unsqueeze(0)
u2,v2,z2,inf2=APP._project(cam,w2)
print("pt2 _project u=%.3f v=%.3f z=%.5f (expect u~%.1f v~%.1f)"%(float(u2),float(v2),float(z2),960+0.1*1956.0875/1,540+(-0.05)*1651.9852/1))
Xb2=COM.backproject(cam,u2,v2,z2)
print("pt2 backproject roundtrip err=%.6f"%float((Xb2-w2).norm()))

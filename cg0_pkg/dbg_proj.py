import sys
sys.path.insert(0, "/root/epic05_n1"); sys.path.insert(0, "/root/epic05_frozen/higs_c0_final30k_worktree"); sys.path.insert(0, "/root/epic05_build_cache/scene/gsplat_scene_cuda"); sys.path.insert(0, "/root/3dgs-renderer-benchmark/scripts/epic05/phase7"); sys.path.insert(0, "/root/3dgs-renderer-benchmark/baseline/reference_v1")
import torch, numpy as np
import n1_core as C
import cg0_prereg as P
import cg0_compiler as COM
import cg0_appearance as APP
from cg0_run import render_depth
C.bootstrap_c0()
scene="room"
ds=C.load_scene_dataset(scene)
cam=ds.get_camera(0)
print("cam0 K=",cam.K.detach().cpu().numpy())
print("cam0 viewmatrix=\n",cam.viewmatrix.detach().cpu().numpy())
# take a synthetic point 5m in front of camera center in camera frame
Ccenter=cam.camera_center
# camera looks down -z in cam frame; a point at cam-frame (0,0,-5) -> world
R=cam.viewmatrix[:3,:3]; t=cam.viewmatrix[:3,3]
# world = R^T (cam_pt - t); cam_pt=(0,0,-5)
cam_pt=torch.tensor([0.0,0.0,-5.0],device="cuda")
world=(cam_pt.unsqueeze(0)-t.unsqueeze(0))  # = R^T? no: cam_pt = R@world + t => world = R^T(cam_pt - t)
world = torch.linalg.solve(R, cam_pt-t.unsqueeze(0)).unsqueeze(0)
print("world point (cam-frame 0,0,-5) =",world.detach().cpu().numpy())
# compiler _project
u,v,z,inf=APP._project(cam,world)
print("_project: u=%.3f v=%.3f z=%.5f inf=%s"%(float(u),float(v),float(z),bool(inf)))
# expected: principal point cx,cy at z=5
# backproject roundtrip via compiler
Xb=COM.backproject(cam,u,v,z)
print("backproject roundtrip err=%.6f"%float((Xb-world).norm()))

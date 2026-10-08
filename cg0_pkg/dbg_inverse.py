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
ck=C.load_checkpoint(scene,it); ss=float(ck["spatial_lr_scale"])
src=C.build_model(ck)
cam=ds.get_camera(0)
R=cam.viewmatrix[:3,:3]; t=cam.viewmatrix[:3,3]; Ccenter=cam.camera_center
xyz=ck["xyz"]
# pick 5 source points, project, then backproject with both variants
for k in [0, 100000, 400000, 700000, 841780]:
    X=xyz[k:k+1].cuda()
    u,v,z,inf=APP._project(cam,X)
    cam_pt=(X[0]@R.t()+t)  # = _project internal
    # code backproject
    Rt=R.t()
    X_code=(cam_pt@Rt)+Ccenter
    # correct inverse: cam_pt @ R + Ccenter  (== R^T(cam_pt - t))
    X_corr=(cam_pt@R)+Ccenter
    d_code=(X_code-X).norm().item(); d_corr=(X_corr-X).norm().item()
    print(f"k={k} u={u.item():.1f} v={v.item():.1f} z={z.item():.3f} inf={inf.item()} | d_code={d_code:.4f} d_corr={d_corr:.4f}")
# also check camera_center == -R^T t
print("Ccenter", Ccenter.tolist(), "-R^T t", (-R.t()@t).tolist())


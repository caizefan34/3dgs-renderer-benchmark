# VOS0 self-contained library: PLY load, camera build, rasterizer + visibility capture.
import json, math, time
import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

PROPS = (["x","y","z","nx","ny","nz","f_dc_0","f_dc_1","f_dc_2"]
         + [f"f_rest_{i}" for i in range(45)]
         + ["opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"])

def load_ply(path):
    with open(path,"rb") as f:
        header=b""
        while b"end_header" not in header:
            header+=f.readline()
        n=int([l for l in header.decode().splitlines() if l.startswith("element vertex")][0].split()[-1])
        buf=f.read(n*len(PROPS)*4)
    arr=np.frombuffer(buf,dtype=np.float32).reshape(n,len(PROPS))
    col={p:arr[:,i] for i,p in enumerate(PROPS)}
    xyz=np.stack([col["x"],col["y"],col["z"]],1).astype(np.float32)
    f_dc=np.stack([col["f_dc_0"],col["f_dc_1"],col["f_dc_2"]],1).astype(np.float32)
    f_rest=np.stack([col[f"f_rest_{i}"] for i in range(45)],1).astype(np.float32)
    opacity=col["opacity"].astype(np.float32)[:,None]
    scaling=np.stack([col["scale_0"],col["scale_1"],col["scale_2"]],1).astype(np.float32)
    rotation=np.stack([col["rot_0"],col["rot_1"],col["rot_2"],col["rot_3"]],1).astype(np.float32)
    return dict(xyz=xyz,f_dc=f_dc,f_rest=f_rest,opacity=opacity,scaling=scaling,rotation=rotation,n=n)

def build_camera(c, device="cuda", znear=0.01, zfar=100.0):
    W2C=np.eye(4,dtype=np.float32)
    W2C[:3,:3]=np.asarray(c["rotation"],dtype=np.float32)
    W2C[:3,3]=np.asarray(c["position"],dtype=np.float32)
    Rt=np.linalg.inv(W2C)
    R=Rt[:3,:3].T.copy(); T=Rt[:3,3].copy()
    fx=c["fx"]; fy=c["fy"]; w=c["width"]; h=c["height"]
    FoVx=2*math.atan(w/(2*fx)); FoVy=2*math.atan(h/(2*fy))
    world_view=torch.tensor(Rt,dtype=torch.float32,device=device).transpose(0,1).contiguous()
    tanfovx=math.tan(FoVx*0.5); tany=math.tan(FoVy*0.5)
    top=tany*znear; bottom=-top; right=tanfovx*znear; left=-right
    P=torch.zeros(4,4,dtype=torch.float32)
    P[0,0]=2.0*znear/(right-left); P[1,1]=2.0*znear/(top-bottom)
    P[0,2]=(right+left)/(right-left); P[1,2]=(top+bottom)/(top-bottom)
    P[3,2]=1.0; P[2,2]=zfar/(zfar-znear); P[2,3]=-(zfar*znear)/(zfar-znear)
    proj=P.to(device).transpose(0,1).contiguous()
    full=(world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0).contiguous()
    cam_center=world_view.inverse()[3,:3]
    return dict(world_view=world_view,proj=proj,full=full,cam_center=cam_center,
                FoVx=FoVx,FoVy=FoVy,W=w,H=h,fx=fx,fy=fy,name=c.get("img_name",str(c.get("id"))))

def shs_from(f_dc,f_rest,active_degree=3):
    f_dc=torch.as_tensor(f_dc,dtype=torch.float32)
    f_rest=torch.as_tensor(f_rest,dtype=torch.float32)
    N=f_dc.shape[0]
    dc=f_dc.reshape(N,3,1); rest=f_rest.reshape(N,3,15)
    return torch.cat([dc,rest],dim=2).to("cuda")

def render(cam, gs, active_degree=3, bg=(0,0,0), raw_params=None, use_precomp=False):
    # raw_params: optional dict of RAW param tensors (requires_grad) to use instead of
    # building from numpy gs. Keys: xyz,f_dc,f_rest,opacity,scaling,rotation.
    if raw_params is not None:
        xyz=raw_params["xyz"]
        opacity=torch.sigmoid(raw_params["opacity"])
        scales=torch.exp(raw_params["scaling"])
        rots=torch.nn.functional.normalize(raw_params["rotation"],dim=1)
        if use_precomp:
            shs=None; colors_precomp=raw_params["f_dc"].reshape(-1,3)+0.5
        else:
            shs=shs_from(raw_params["f_dc"],raw_params["f_rest"],active_degree); colors_precomp=None
    else:
        xyz=torch.tensor(gs["xyz"],device="cuda",dtype=torch.float32)
        shs=shs_from(gs["f_dc"],gs["f_rest"],active_degree); colors_precomp=None
        opacity=torch.sigmoid(torch.tensor(gs["opacity"],device="cuda",dtype=torch.float32))
        scales=torch.exp(torch.tensor(gs["scaling"],device="cuda",dtype=torch.float32))
        rots=torch.nn.functional.normalize(torch.tensor(gs["rotation"],device="cuda",dtype=torch.float32),dim=1)
    screenspace=torch.zeros_like(xyz,requires_grad=True)
    bgt=torch.tensor(bg,device="cuda",dtype=torch.float32)
    _fields=GaussianRasterizationSettings._fields if hasattr(GaussianRasterizationSettings,"_fields") else None
    _kw=dict(image_height=int(cam["H"]),image_width=int(cam["W"]),
        tanfovx=math.tan(cam["FoVx"]*0.5),tanfovy=math.tan(cam["FoVy"]*0.5),
        bg=bgt,scale_modifier=1.0,viewmatrix=cam["world_view"],
        projmatrix=cam["full"],sh_degree=active_degree,
        campos=cam["cam_center"],prefiltered=False,debug=False)
    if _fields is None or "antialiasing" in _fields:
        _kw["antialiasing"]=False
    rs=GaussianRasterizationSettings(**_kw)
    rz=GaussianRasterizer(raster_settings=rs)
    out=rz(means3D=xyz,means2D=screenspace,shs=shs,colors_precomp=colors_precomp,
        opacities=opacity,scales=scales,rotations=rots,cov3D_precomp=None)
    img=out[0]; radii=out[1]; depth=out[2] if len(out)>2 else None
    return dict(img=img,radii=radii,depth=depth,screenspace=screenspace,
                visibility=(radii>0),xyz=xyz,shs=shs,opacity=opacity,scales=scales,rots=rots)

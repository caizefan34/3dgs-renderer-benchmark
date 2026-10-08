#!/bin/bash
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY - <<'PYEOF'
import struct, numpy as np, torch, gsplat
D='/tmp/3dgs-renderer-benchmark/data'
dev='cuda'
def read_cameras(path):
    with open(path,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; out={}
        for _ in range(n):
            cid=struct.unpack('<i',f.read(4))[0]; mid=struct.unpack('<i',f.read(4))[0]
            w=struct.unpack('<Q',f.read(8))[0]; h=struct.unpack('<Q',f.read(8))[0]
            npar={0:3,1:4,2:3,3:4,4:4,5:5,6:6,7:8}.get(mid,4)
            p=struct.unpack(f'<{npar}d',f.read(npar*8)); out[cid]=(mid,w,h,p)
    return out
def read_images(path):
    with open(path,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; out=[]
        for _ in range(n):
            iid=struct.unpack('<i',f.read(4))[0]; qv=struct.unpack('<4d',f.read(32)); tv=struct.unpack('<3d',f.read(24)); cam=struct.unpack('<i',f.read(4))[0]
            name=b''
            while True:
                c=f.read(1)
                if c==b'\x00': break
                name+=c
            np_=struct.unpack('<Q',f.read(8))[0]; f.read(np_*24)
            out.append((iid,name.decode(),cam,qv,tv))
    return out
def q2R(q):
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
s='room'
cams=read_cameras(f'{D}/datasets/mipnerf360/{s}/sparse/0/cameras.bin')
imgs=read_images(f'{D}/datasets/mipnerf360/{s}/sparse/0/images.bin')
ck=torch.load('/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_room_30k/checkpoints/iter_30000.pt',map_location='cpu',weights_only=False)
xyz=ck['xyz'].numpy(); scaling=torch.exp(ck['scaling'].numpy()).astype(np.float32); rotation=ck['rotation'].numpy().astype(np.float32); opacity=torch.sigmoid(ck['opacity'].numpy()).astype(np.float32)
shs=ck['shs'].numpy()
print('n',len(xyz),'sh_degree',ck['active_sh_degree'])
# pick view 0
iid,name,cam,qv,tv=imgs[0]
mid,W,H,par=cams[cam]
fx,fy,cx,cy=par
R_w2c=q2R(qv); t_w2c=np.array(tv)
R_c2w=R_w2c.T; t_c2w=-R_w2c.T@t_w2c
viewmat=torch.as_tensor(np.eye(4,dtype=np.float32),dtype=torch.float32,device=dev)
viewmat[:3,:3]=torch.as_tensor(R_w2c,dtype=torch.float32,device=dev); viewmat[:3,3]=torch.as_tensor(t_w2c,dtype=torch.float32,device=dev)
K=torch.as_tensor([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=torch.float32,device=dev)[None]
campos=torch.as_tensor(t_c2w,dtype=torch.float32,device=dev)[None]
print('view',name,'W',W,'H',H,'campos',campos.cpu().numpy())
means=torch.as_tensor(xyz,dtype=torch.float32,device=dev)
scales=torch.as_tensor(scaling,dtype=torch.float32,device=dev)
quats=torch.as_tensor(rotation,dtype=torch.float32,device=dev)
opas=torch.as_tensor(opacity,dtype=torch.float32,device=dev)
m2d,depths,conics,radii,comp=gsplat.fully_fused_projection(means,covars=None,quats=quats,scales=scales,viewmats=viewmat,Ks=K,width=W,height=H,eps2d=0.3,near_plane=0.01,far_plane=1e10,radius_clip=0.0,packed=False,calc_compensations=False)
print('means2d',tuple(m2d.shape),'depths',tuple(depths.shape),'conics',tuple(conics.shape),'radii',tuple(radii.shape))
valid=(radii[...,0]>0)
print('valid gaussians',int(valid.sum()),'/',len(xyz))
# tile structure
TILE=16
tw=(W+TILE-1)//TILE; th=(H+TILE-1)//TILE
print('tiles',tw,'x',th,'=',tw*th)
tile_ids,gaussian_ids,_=gsplat.isect_tiles(m2d,radii,depths,TILE,tw,th,sort=True)
print('tile_ids',tuple(tile_ids.shape),'gaussian_ids',tuple(gaussian_ids.shape),'n_intersections',len(gaussian_ids))
# per-tile gaussian counts
import collections
cnt=torch.bincount(tile_ids,minlength=tw*th)
cnt=cnt[cnt>0]
print('nonempty tiles',len(cnt),'mean per tile',float(cnt.float().mean()),'median',float(cnt.float().median()),'p99',float(cnt.float().quantile(0.99)),'max',int(cnt.max()))
PYEOF

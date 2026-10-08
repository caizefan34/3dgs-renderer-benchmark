#!/bin/bash
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY - <<'PYEOF'
import struct, numpy as np, torch, gsplat
D='/tmp/3dgs-renderer-benchmark/data'; dev='cuda'
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
means=ck['xyz'].to(dev).float(); scales=torch.exp(ck['scaling'].to(dev).float()); quats=ck['rotation'].to(dev).float(); opas=torch.sigmoid(ck['opacity'].to(dev).float()); shs=ck['shs'].to(dev).float()
iid,name,cam,qv,tv=imgs[0]; mid,W,H,par=cams[cam]; fx,fy,cx,cy=par
R_w2c=q2R(qv); t_w2c=np.array(tv); t_c2w=-R_w2c.T@t_w2c
viewmat=torch.eye(4,dtype=torch.float32,device=dev)[None]; viewmat[0,:3,:3]=torch.as_tensor(R_w2c,dtype=torch.float32,device=dev); viewmat[0,:3,3]=torch.as_tensor(t_w2c,dtype=torch.float32,device=dev)
K=torch.as_tensor([[[fx,0,cx],[0,fy,cy],[0,0,1]]],dtype=torch.float32,device=dev)
campos=torch.as_tensor(t_c2w,dtype=torch.float32,device=dev)[None]
render_colors,render_alphas,meta=gsplat.rasterization(means,quats,scales,opas,shs,viewmat,K,width=W,height=H,sh_degree=3,render_mode='RGB',packed=True,tile_size=16)
print('render_colors',tuple(render_colors.shape),'alphas',tuple(render_alphas.shape))
print('meta keys:',sorted(meta.keys()))
for k in ['gaussian_ids','tile_offsets','width','height','n_cameras','radii','mean2d','conics','depths','tile_width','tile_height']:
    if k in meta: print(' ',k,tuple(meta[k].shape) if hasattr(meta[k],'shape') else meta[k])
gi=meta['gaussian_ids']; to=meta['tile_offsets']
print('n_gaussian_ids',len(gi),'n_tiles',len(to)-1)
cnt=torch.diff(to)
print('per-tile counts: nonempty',int((cnt>0).sum()),'mean',float(cnt[cnt>0].float().mean()),'median',float(cnt[cnt>0].float().median()),'p99',float(cnt[cnt>0].float().quantile(0.99)),'max',int(cnt.max()))
PYEOF

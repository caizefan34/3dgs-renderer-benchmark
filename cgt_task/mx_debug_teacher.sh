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
cams=read_cameras(f'{D}/datasets/mipnerf360/room/sparse/0/cameras.bin')
imgs=read_images(f'{D}/datasets/mipnerf360/room/sparse/0/images.bin')
ck=torch.load('/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_room_30k/checkpoints/iter_30000.pt',map_location='cpu',weights_only=False)
means=ck['xyz'].to(dev).float(); scales=torch.exp(ck['scaling'].to(dev).float()); quats=ck['rotation'].to(dev).float(); opacities=torch.sigmoid(ck['opacity'].to(dev).float()); shs=ck['shs'].to(dev).float()
iid,name,cam,qv,tv=imgs[0]; mid,W,H,par=cams[cam]; fx,fy,cx,cy=par
R_w2c=q2R(qv); t_w2c=np.array(tv); t_c2w=-R_w2c.T@t_w2c
viewmat=torch.eye(4,dtype=torch.float32,device=dev)[None]; viewmat[0,:3,:3]=torch.as_tensor(R_w2c,dtype=torch.float32,device=dev); viewmat[0,:3,3]=torch.as_tensor(t_w2c,dtype=torch.float32,device=dev)
K=torch.as_tensor([[[fx,0,cx],[0,fy,cy],[0,0,1]]],dtype=torch.float32,device=dev)
campos=torch.as_tensor(t_c2w,dtype=torch.float32,device=dev)[None]
rc,ra,meta=gsplat.rasterization(means,quats,scales,opacities,shs,viewmat,K,width=W,height=H,sh_degree=3,render_mode='RGB',packed=True,tile_size=16)
dirs=torch.nn.functional.normalize(means-campos,dim=-1)
colors=(gsplat.spherical_harmonics(3,dirs,shs)+0.5).clamp(0,1).cpu().numpy()
fi=meta['flatten_ids'].cpu().numpy(); gi=meta['gaussian_ids'].cpu().numpy()
print('fi range',fi.min(),fi.max(),'gi range',gi.min(),gi.max(),'n_visible',len(gi))
# pick tile with ~200 gaussians
io=meta['isect_offsets'].cpu().numpy().reshape(-1); th,tw=meta['tile_height'],meta['tile_width']
def end_of(ti,tj):
    if tj+1<tw: return int(io[ti*tw+tj+1])
    if ti+1<th: return int(io[(ti+1)*tw])
    return len(fi)
# find a mid-size tile
best=None
for ti in range(th):
    for tj in range(tw):
        c=end_of(ti,tj)-int(io[ti*tw+tj])
        if c>50 and c<300: best=(ti,tj); break
    if best: break
ti,tj=best; s=int(io[ti*tw+tj]); e=end_of(ti,tj); gids=fi[s:e]
print('tile',ti,tj,'n',e-s,'gids sample',gids[:5])
means2d=meta['means2d'].cpu().numpy(); conics=meta['conics'].cpu().numpy(); opac=meta['opacities'].cpu().numpy()
print('means2d[gids]',means2d[gids].shape,'conics[gids]',conics[gids].shape,'opac[gids]',opac[gids].shape,'colors[gids]',colors[gids].shape)
# now run teacher
TILE=16; ox=tj*TILE; oy=ti*TILE
xs=ox+np.arange(TILE)+0.5; ys=oy+np.arange(TILE)+0.5
gx,gy=np.meshgrid(xs,ys,indexing='xy')
px=np.stack([gx.ravel(),gy.ravel()],axis=1).astype(np.float64)
cx0,cy0=px.mean(0); px[:,0]-=cx0; px[:,1]-=cy0
P=px.shape[0]; m=len(gids)
mu=means2d[gids]; co=conics[gids]; op=opac[gids]; col=colors[gids]
print('P',P,'m',m,'mu',mu.shape,'co',co.shape,'op',op.shape,'col',col.shape)
dx=px[:,0:1]-mu[None,:,0]; dy=px[:,1:2]-mu[None,:,1]
print('dx',dx.shape)
a=co[:,0]; b=co[:,1]; c=co[:,2]
power=-0.5*(a[None,:]*dx*dx+2*b[None,:]*dx*dy+c[None,:]*dy*dy)
print('power',power.shape)
alpha=np.where(power>-4.5, op[None,:]*np.exp(np.clip(power,None,0)),0.0)
print('alpha',alpha.shape)
alpha=np.clip(alpha,0,0.999)
Tprefix=np.cumprod(1-alpha,axis=1)
Tbefore=np.concatenate([np.ones((P,1)),Tprefix[:,:-1]],axis=1)
E=np.einsum('pi,pi,ic->pc',Tbefore,alpha,col)
T=Tprefix[:,-1:]
print('E',E.shape,'T',T.shape)
print('E range',E.min(),E.max(),'T range',T.min(),T.max())
PYEOF

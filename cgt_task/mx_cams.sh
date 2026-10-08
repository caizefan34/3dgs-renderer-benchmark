#!/bin/bash
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY - <<'PYEOF'
import struct, numpy as np, torch, gsplat
D='/tmp/3dgs-renderer-benchmark/data'
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
            iid=struct.unpack('<i',f.read(4))[0]; qv=struct.unpack('<4d',f.read(32))
            tv=struct.unpack('<3d',f.read(24)); cam=struct.unpack('<i',f.read(4))[0]
            name=b''
            while True:
                c=f.read(1)
                if c==b'\x00': break
                name+=c
            # 2D points
            np_=struct.unpack('<Q',f.read(8))[0]
            f.read(np_*24)
            out.append((iid,name.decode(),cam,qv,tv))
    return out
def q2R(q):
    x,y,z,w=q
    R=np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return R
for s in ['room','garden','bicycle']:
    cams=read_cameras(f'{D}/datasets/mipnerf360/{s}/sparse/0/cameras.bin')
    imgs=read_images(f'{D}/datasets/mipnerf360/{s}/sparse/0/images.bin')
    print(s,'cameras',len(cams),'images',len(imgs),'first',imgs[0][1],'model',cams[imgs[0][2]][0],'WxH',cams[imgs[0][2]][1],cams[imgs[0][2]][2])
PYEOF

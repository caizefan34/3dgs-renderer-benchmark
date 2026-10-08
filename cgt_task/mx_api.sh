#!/bin/bash
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY - <<'PYEOF'
import inspect, torch, gsplat
print("=== fully_fused_projection sig ===")
print(inspect.signature(gsplat.fully_fused_projection))
print("=== rasterization sig ===")
print(inspect.signature(gsplat.rasterization))
# load room 30k
ck=torch.load('/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_room_30k/checkpoints/iter_30000.pt',map_location='cpu',weights_only=False)
print("n_gaussians",ck['num_points'],"sh_degree",ck['active_sh_degree'])
# read one camera from cameras.bin (COLMAP)
try:
    from plyfile import PlyData
except Exception as e:
    print("plyfile err",e)
import struct
def read_cameras_bin(path):
    cams={}
    with open(path,'rb') as f:
        (num,)=struct.unpack('<Q',f.read(8))
        for _ in range(num):
            (cid,model,w,h,params)=struct.unpack('<iiQQId',f.read(8+8+8+8+8+8+8))[:7]
            cams[cid]=(model,w,h,params)
    return cams
try:
    cams=read_cameras_bin('/tmp/3dgs-renderer-benchmark/data/datasets/mipnerf360/room/sparse/0/cameras.bin')
    print("cameras",len(cams),"first",list(cams.items())[0])
except Exception as e:
    print("cameras.bin read err",repr(e))
PYEOF

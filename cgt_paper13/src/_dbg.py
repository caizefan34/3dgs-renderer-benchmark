import sys, os, numpy as np, ctypes, faulthandler, time
faulthandler.enable()
sys.path.insert(0, "/home/liaoyuanjun/cgt_realscene/final/SCRIPTS")
sys.path.insert(0, "/home/liaoyuanjun/cgt_f0_fused_cuda/src")
from g1_compressibility import tile_teacher, fit_cgt, TILE
from g2_rigorous_certificate import certify_tile, extract_tiles
R=4; NPX=256
def shared_px():
    xs=np.arange(TILE)+0.5; ys=np.arange(TILE)+0.5
    gx,gy=np.meshgrid(xs,ys,indexing="xy")
    px=np.stack([gx.ravel(),gy.ravel()],axis=1).astype(np.float64)
    c=px.mean(0); px[:,0]-=c[0]; px[:,1]-=c[1]
    return np.ascontiguousarray(px)
px=shared_px()
scene="room"; ckpt="/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_room_multiphase_v2/checkpoints/iter_15000.pt"
all_tiles=[]
for vi in range(2):
    all_tiles+=extract_tiles(scene, ckpt, "/tmp/3dgs-renderer-benchmark/data", vi, 5, 20261004+vi)
print(f"total {len(all_tiles)} tiles", flush=True)
rng=np.random.default_rng(20261004)
m2l,col_,opl,cll,cfl=[],[],[],[],[]; safel=[]; Ml=[]; off=[0]
t0=time.perf_counter()
for ti,t in enumerate(all_tiles):
    gids=t["gids"]; m2=t["means2d"]; con=t["conics"]; op=t["opac"]; cl=t["colors"]
    M=int(gids.shape[0])
    E,T=tile_teacher(px[:,0:1],px[:,1:2],gids,m2,con,op,cl)
    y=np.concatenate([E,T],axis=1); P=y.shape[0]
    tr=np.arange(0,P,2); yt=y[tr]; qt=np.column_stack([px[tr,0],px[tr,1]]).astype(np.float64)
    p=fit_cgt(qt,yt,R,20261004+ti*1000+R,200)
    res=certify_tile(t,p,1.0,10,rng,True)
    safel.append(int(res["passed"])); Ml.append(M)
    cfl.append(np.ascontiguousarray(p,np.float64).reshape(R,10))
    m2l.append(np.ascontiguousarray(m2[gids],np.float64)); col_.append(np.ascontiguousarray(con[gids],np.float64))
    opl.append(np.ascontiguousarray(op[gids],np.float64)); cll.append(np.ascontiguousarray(cl[gids],np.float64))
    off.append(off[-1]+M)
    print(f"tile{ti} M={M} safe={res['passed']} rig={res['rigorous_max_abs']:.4f} nan={np.isnan(p).any()} t={time.perf_counter()-t0:.1f}s", flush=True)
Mt=off[-1]; n=len(all_tiles)
print(f"Mt={Mt} n={n} building arrays...", flush=True)
means2d=np.ascontiguousarray(np.concatenate(m2l,0),np.float64)
conics=np.ascontiguousarray(np.concatenate(col_,0),np.float64)
opac=np.ascontiguousarray(np.concatenate(opl,0),np.float64)
colors=np.ascontiguousarray(np.concatenate(cll,0),np.float64)
coeffs=np.ascontiguousarray(np.stack(cfl,0),np.float64).reshape(n,R,10)
safe=np.asarray(safel,np.int32); offa=np.asarray(off,np.int32)
print(f"arrays: m2={means2d.shape} co={conics.shape} op={opac.shape} cl={colors.shape} cf={coeffs.shape} off={offa.shape}", flush=True)
lib=ctypes.CDLL("/home/liaoyuanjun/cgt_f0_fused_cuda/src/f0_kernels.so")
P_=ctypes.c_void_p; I=ctypes.c_int; S=ctypes.c_size_t
lib.f0_init(I(1))
def up(a):
    a=np.ascontiguousarray(a); d=lib.f0_malloc(S(a.nbytes)); lib.f0_h2d(P_(d),a.ctypes.data_as(P_),S(a.nbytes)); return int(d)
d_m2=up(means2d); d_co=up(conics); d_op=up(opac); d_cl=up(colors); d_px=up(px); d_cf=up(coeffs); d_off=up(offa); d_safe=up(safe)
d_out=up(np.zeros((n,NPX,4),np.float32))
LF=[P_,P_,P_,P_,P_,P_,I,P_,P_,P_,I]
lib.f0_fused_f32.argtypes=LF; lib.f0_teacher_f32.argtypes=LF; lib.f0_stream_sync.argtypes=[]
print("launching fused on all tiles...", flush=True)
lib.f0_fused_f32(P_(d_m2),P_(d_co),P_(d_op),P_(d_cl),P_(d_off),P_(d_cf),I(R),P_(d_safe),P_(d_px),P_(d_out),I(n))
lib.f0_stream_sync()
print("fused done; launching teacher...", flush=True)
lib.f0_teacher_f32(P_(d_m2),P_(d_co),P_(d_op),P_(d_cl),P_(d_off),P_(d_px),P_(d_out),I(n))
lib.f0_stream_sync()
h=np.zeros((n,NPX,4),np.float32); lib.f0_d2h(h.ctypes.data_as(P_),P_(d_out),S(h.nbytes))
print(f"OUT max={np.abs(h).max():.4f} nan={np.isnan(h).any()}  DEBUG_DONE", flush=True)
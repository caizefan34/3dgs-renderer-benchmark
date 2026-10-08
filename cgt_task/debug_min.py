import sys, numpy as np
sys.path.insert(0, r"C:\Users\36570\3dgs-renderer-benchmark\cgt_realscene_results\final\SCRIPTS")
import g2_rigorous_certificate as G
from g1_compressibility import tile_teacher
rng = np.random.default_rng(7)
def rand_tile(rng, M, TILE=16):
    xs=np.arange(TILE)+0.5; ys=np.arange(TILE)+0.5
    gx,gy=np.meshgrid(xs,ys,indexing="xy"); px=np.stack([gx.ravel(),gy.ravel()],1).astype(float)
    c=px.mean(0); px[:,0]-=c[0]; px[:,1]-=c[1]
    means2d=rng.uniform(-8,8,size=(M,2)); conics=np.zeros((M,3))
    conics[:,0]=rng.uniform(0.5,2,M); conics[:,2]=rng.uniform(0.5,2,M); conics[:,1]=rng.uniform(-0.5,0.5,M)
    opac=rng.uniform(0.05,0.99,M); colors=rng.uniform(0,1,size=(M,3)); gids=np.arange(M)
    return dict(px=px,gids=gids,means2d=means2d,conics=conics,opac=opac,colors=colors,M=M)
def rand_student(rng,R):
    p=np.zeros((R,10)); p[:,0]=rng.uniform(-8,8,R); p[:,1]=rng.uniform(-8,8,R); p[:,2]=rng.uniform(-2,2,R)
    p[:,3]=rng.uniform(-2,2,R); p[:,4]=rng.uniform(-2,2,R); p[:,5]=rng.uniform(-3,3,R); p[:,6]=rng.uniform(-3,3,R)
    p[:,7:10]=rng.uniform(-3,3,R*3).reshape(R,3); return p
for trial in range(40):
    M=int(rng.integers(3,120)); R=int(rng.choice([4,8,16]))
    t=rand_tile(rng,M); p=rand_student(rng,R)
    if trial==13:
        ctx=(t["gids"],t["means2d"],t["conics"],t["opac"],t["colors"],p)
        gids,means2d,conics,opac,cl=ctx[:5]
        sx,sy=5.102559813303726,1.0354529095703415
        # point teacher via tile_teacher
        E,T=tile_teacher(np.array([[sx]]),np.array([[sy]]),*ctx[:5])
        print("tile_teacher E,T:",E[0],T[0,0])
        # manual per-g alpha at point (tile_teacher formula)
        mu=means2d[gids]; co=conics[gids]; a,b,c=co[:,0],co[:,1],co[:,2]
        dx=sx-mu[:,0]; dy=sy-mu[:,1]
        pw=-0.5*(a*dx*dx+2*b*dx*dy+c*dy*dy)
        pa=np.where(pw>G.POWER_CLIP,opac*np.exp(np.clip(pw,None,0)),0.0); pa=np.clip(pa,0,0.999)
        print("manual per-g alpha:",pa)
        print("manual power:",pw)
        # teacher_interval at box
        box=[4.6875,5.625,0.9375,1.875]
        tl=G.teacher_interval((box[0],box[1]),(box[2],box[3]),*ctx[:5])
        print("teacher_interval lo,hi:",tl)
        # interval per-g alpha bounds
        mu2=means2d[gids]; co2=conics[gids]; a2,b2,c2=co2[:,0],co2[:,1],co2[:,2]
        dxi=(box[0]-mu2[:,0],box[1]-mu2[:,0]); dyi=(box[2]-mu2[:,1],box[3]-mu2[:,1])
        s=G.iadd(G.iadd(G.imul((a2,a2),G.isqrt(dxi)),G.imul((2*b2,2*b2),G.imul(dxi,dyi))),G.imul((c2,c2),G.isqrt(dyi)))
        power=G.neg_half(s)
        active=(np.clip(opac*np.exp(np.minimum(power[0],0)),0,0.999),np.clip(opac*np.exp(np.minimum(power[1],0)),0,0.999))
        ahi=np.where(power[1]<=G.POWER_CLIP,0.0,active[1]); alo=np.where(power[1]<=G.POWER_CLIP,0.0,np.where(power[0]>G.POWER_CLIP,active[0],0.0))
        print("interval ahi:",ahi)
        print("interval alo:",alo)
        print("interval power_lo,hi:",power)
        print("point power in [lo,hi]?",np.all((pw>=power[0]-1e-9)&(pw<=power[1]+1e-9)))
        print("point alpha in [alo,ahi]?", np.all((pa>=alo-1e-9)&(pa<=ahi+1e-9)))
        break

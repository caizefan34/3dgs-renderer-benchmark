import sys, numpy as np
sys.path.insert(0, r"C:\Users\36570\3dgs-renderer-benchmark\cgt_realscene_results\final\SCRIPTS")
import g2_rigorous_certificate as G
import importlib
g1 = importlib.import_module("g1_compressibility")

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
        box=[4.6875,5.625,0.9375,1.875]; px0,px1,py0,py1=box
        gids,means2d,conics,opac,cl=ctx[:5]
        mu=means2d[gids]; co=conics[gids]; op=opac[gids]
        a,b,cc=co[:,0],co[:,1],co[:,2]
        # per-gaussian point alpha at a scan of pixels; find max alpha
        dxlo=px0-mu[:,0]; dxhi=px1-mu[:,0]; dylo=py0-mu[:,1]; dyhi=py1-mu[:,1]
        # interval power
        dxi=(dxlo,dxhi); dyi=(dylo,dyhi)
        s=G.iadd(G.iadd(G.imul((a,a),G.isqrt(dxi)),G.imul((2*b,2*b),G.imul(dxi,dyi))),G.imul((cc,cc),G.isqrt(dyi)))
        power=G.neg_half(s)
        active=(np.clip(op*np.exp(np.minimum(power[0],0)),0,0.999),np.clip(op*np.exp(np.minimum(power[1],0)),0,0.999))
        ahi=np.where(power[1]<=G.POWER_CLIP,0.0,active[1]); alo=np.where(power[1]<=G.POWER_CLIP,0.0,np.where(power[0]>G.POWER_CLIP,active[0],0.0))
        # point alpha at center of box
        cx=0.5*(px0+px1); cy=0.5*(py0+py1)
        dxc=cx-mu[:,0]; dyc=cy-mu[:,1]
        pw=-0.5*(a*dxc*dxc+2*b*dxc*dyc+cc*dyc*dyc)
        pt_alpha=np.where(pw>G.POWER_CLIP,op*np.exp(np.clip(pw,None,0)),0.0); pt_alpha=np.clip(pt_alpha,0,0.999)
        bad=np.where(pt_alpha>ahi+1e-9)[0]
        print("box",box,"M",M)
        print("num gauss with point_alpha>ahi:",len(bad))
        for k in bad[:8]:
            print(f"  g{k}: pt_alpha={pt_alpha[k]:.5f} ahi={ahi[k]:.5f} alo={alo[k]:.5f} pw={pw[k]:.4f} power_lo={power[0][k]:.4f} power_hi={power[1][k]:.4f} active=({active[0][k]:.5f},{active[1][k]:.5f})")
        break

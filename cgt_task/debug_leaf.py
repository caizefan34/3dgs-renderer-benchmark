import sys, numpy as np
sys.path.insert(0, r"C:\Users\36570\3dgs-renderer-benchmark\cgt_realscene_results\final\SCRIPTS")
import g2_rigorous_certificate as G
from g1_compressibility import tile_teacher, predict_and_vjp

rng = np.random.default_rng(7)
# advance rng to trial 13 state exactly as validate does
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
        x0,x1=float(t["px"][:,0].min()),float(t["px"][:,0].max()); y0,y1=float(t["px"][:,1].min()),float(t["px"][:,1].max())
        # manual B&B dumping first invalid leaf
        def rec(px0,px1,py0,py1,depth):
            bound,(lo,hi)=G.box_error_bound((px0,px1),(py0,py1),ctx)
            w=px1-px0;h=py1-py0
            if bound<=G.GATE or w<=1.0 or h<=1.0 or depth>=9:
                # validate
                bnd=max(float(np.max(np.abs(lo))),float(np.max(np.abs(hi))))
                for _ in range(8):
                    sx=rng.uniform(px0,px1); sy=rng.uniform(py0,py1); pe=G.point_error(sx,sy,ctx)
                    if pe>bnd+1e-9:
                        print("INVALID LEAF depth",depth,"box",[px0,px1,py0,py1],"bound",bnd,"point_err",pe)
                        tl=G.teacher_interval((px0,px1),(py0,py1),*ctx[:5]); sl=G.student_interval((px0,px1),(py0,py1),ctx[-1])
                        print(" teacher lo",tl[0],"hi",tl[1])
                        print(" student lo",sl[0],"hi",sl[1])
                        E,T=tile_teacher(np.array([[sx]]),np.array([[sy]]),*ctx[:5]); ypt=np.concatenate([E,T],1)[0]
                        pred=predict_and_vjp(p,np.array([[sx,sy]],float))[0]
                        print(" point teacher",ypt,"point student",pred)
                        # per-comp check
                        for c_ in range(4):
                            print(f"  comp{c_}: tin {tl[0][c_]:.5f}<= {ypt[c_]:.5f}<={tl[1][c_]:.5f} ? {tl[0][c_]<=ypt[c_]<=tl[1][c_]} | sin {sl[0][c_]:.5f}<={pred[c_]:.5f}<={sl[1][c_]:.5f} ? {sl[0][c_]<=pred[c_]<=sl[1][c_]}")
                        return True
                return False
            mx=0.5*(px0+px1);my=0.5*(py0+py1)
            for b in [(px0,mx,py0,my),(mx,px1,py0,my),(px0,mx,my,py1),(mx,px1,my,py1)]:
                if rec(b[0],b[1],b[2],b[3],depth+1): return True
            return False
        rec(x0,x1,y0,y1,0)
        break

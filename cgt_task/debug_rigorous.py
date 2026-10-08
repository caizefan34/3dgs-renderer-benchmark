import sys, os, numpy as np
sys.path.insert(0, r"C:\Users\36570\3dgs-renderer-benchmark\cgt_realscene_results\final\SCRIPTS")
import g2_rigorous_certificate as G
from scipy.special import expit
sys.path.insert(0, r"C:\Users\36570\3dgs-renderer-benchmark\cgt_realscene_results\final\SCRIPTS")
from g1_compressibility import tile_teacher, predict_and_vjp

rng = np.random.default_rng(7)
# replay to trial 13 (M=11 R=8)
target=None
for trial in range(40):
    M=int(rng.integers(3,120)); R=int(rng.choice([4,8,16]))
    xs=np.arange(16)+0.5; ys=np.arange(16)+0.5
    gx,gy=np.meshgrid(xs,ys,indexing="xy"); px=np.stack([gx.ravel(),gy.ravel()],1).astype(float)
    c=px.mean(0); px[:,0]-=c[0]; px[:,1]-=c[1]
    means2d=rng.uniform(-8,8,size=(M,2)); conics=np.zeros((M,3))
    conics[:,0]=rng.uniform(0.5,2,M); conics[:,2]=rng.uniform(0.5,2,M); conics[:,1]=rng.uniform(-0.5,0.5,M)
    opac=rng.uniform(0.05,0.99,M); colors=rng.uniform(0,1,size=(M,3)); gids=np.arange(M)
    p=np.zeros((R,10)); p[:,0]=rng.uniform(-8,8,R); p[:,1]=rng.uniform(-8,8,R); p[:,2]=rng.uniform(-2,2,R)
    p[:,3]=rng.uniform(-2,2,R); p[:,4]=rng.uniform(-2,2,R); p[:,5]=rng.uniform(-3,3,R); p[:,6]=rng.uniform(-3,3,R)
    p[:,7:10]=rng.uniform(-3,3,R*3).reshape(R,3)
    t=dict(px=px,gids=gids,means2d=means2d,conics=conics,opac=opac,colors=colors,M=M)
    ctx=(gids,means2d,conics,opac,colors,p)
    if trial==13:
        # scan all pixels, find worst violation
        worst=0; wp=None
        for sx in np.linspace(px[:,0].min(),px[:,0].max(),40):
            for sy in np.linspace(px[:,1].min(),px[:,1].max(),40):
                pe=G.point_error(sx,sy,ctx)
                if pe>worst: worst=pe; wp=(sx,sy)
        # full-box interval
        x0,x1=px[:,0].min(),px[:,0].max(); y0,y1=px[:,1].min(),px[:,1].max()
        (bound,(lo,hi))=G.box_error_bound((x0,x1),(y0,y1),ctx); tl=G.teacher_interval((x0,x1),(y0,y1),*ctx[:5]); sl=G.student_interval((x0,x1),(y0,y1),ctx[-1])
        # point values at wp
        E,T=tile_teacher(np.array([[wp[0]]]),np.array([[wp[1]]]),gids,means2d,conics,opac,colors)
        ypt=np.concatenate([E,T],1)[0]
        pred=predict_and_vjp(p,np.array([[wp[0],wp[1]]],float))[0]
        print("M",M,"R",R,"wp",wp)
        print("point teacher E,T:",ypt)
        print("point student     :",pred)
        print("point err         :",np.abs(pred-ypt), "max",worst)
        print("interval teacher lo,hi:",tl)
        print("interval student lo,hi:",sl)
        print("err interval lo,hi:",lo,hi,"bound",bound)
        # check teacher interval validity at wp
        print("teacher in? E0",tl[0][0]<=ypt[0]<=tl[1][0], "E1",tl[0][1]<=ypt[1]<=tl[1][1],"E2",tl[0][2]<=ypt[2]<=tl[1][2],"T",tl[0][3]<=ypt[3]<=tl[1][3])
        print("student in? S0",sl[0][0]<=pred[0]<=sl[1][0],"S1",sl[0][1]<=pred[1]<=sl[1][1],"S2",sl[0][2]<=pred[2]<=sl[1][2],"ST",sl[0][3]<=pred[3]<=sl[1][3])
        break

import sys, numpy as np
sys.path.insert(0, r"C:\Users\36570\3dgs-renderer-benchmark\cgt_realscene_results\final\SCRIPTS")
from g1_compressibility import tile_teacher
# single gaussian at mu=(2,3), conic=(1,1,1), op=0.99, color=(1,1,1)
gids=np.arange(1); means2d=np.array([[2.0,3.0]]); conics=np.array([[1.0,1.0,1.0]]); opac=np.array([0.99]); colors=np.array([[1.0,1.0,1.0]])
for (x,y) in [(2.0,3.0),(2.0,2.0),(5.0,5.0)]:
    E,T=tile_teacher(np.array([[x]]),np.array([[y]]),gids,means2d,conics,opac,colors)
    # correct alpha: dx=x-2, dy=y-3, power=-0.5*(dx^2+2dxdy+dy^2)
    dx=x-2; dy=y-3; pw=-0.5*(dx*dx+2*dx*dy+dy*dy); pa=0.99*np.exp(min(pw,0)) if pw>-4.5 else 0.0
    print(f"pixel({x},{y}): tile_teacher E={E[0]} T={T[0,0]:.4f} | correct alpha={pa:.4f} (peak at (2,3) alpha=0.99)")

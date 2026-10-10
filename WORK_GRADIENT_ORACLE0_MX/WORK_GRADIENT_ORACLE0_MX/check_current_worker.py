import numpy as np, sys
D="/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/gp0_renders"
PK=["means","quats","scales","opac","colors"]
for ct in ["5K"]:
    for ci in [0]:
        b=np.load(f"{D}/{ct}_B_cam{ci}_grad.npz"); c=np.load(f"{D}/{ct}_C_cam{ci}_grad.npz")
        print(f"=== {ct}_cam{ci} (current worker, separate subprocesses) ===")
        for k in PK:
            a=b[k].flatten(); bb=c[k].flatten()
            cos=float(np.dot(a,bb)/(np.linalg.norm(a)*np.linalg.norm(bb)+1e-12))
            print(f"  {k}: cos={cos:.4f} zeroB={float(np.mean(np.isclose(b[k],0))):.3f} zeroC={float(np.mean(np.isclose(c[k],0))):.3f}")

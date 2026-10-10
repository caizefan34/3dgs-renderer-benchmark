import numpy as np, os
D="/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp"
IMG=f"{D}/s1a_r1_deliverables/gp0_renders"
PK=["means","quats","scales","opac","colors"]
for tag in ["5K","10K"]:
    gb=np.load(f"{IMG}/{tag}_B_cam0_grad.npz"); gc=np.load(f"{IMG}/{tag}_C_cam0_grad.npz")
    print(f"=== {tag}_cam0 ===")
    for k in PK:
        a=gb[k]; b=gc[k]
        an=np.isnan(a).sum(); bn=np.isnan(b).sum()
        az=np.isclose(a,0).sum(); bz=np.isclose(b,0).sum()
        c=float(np.dot(a.flatten(),b.flatten())/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
        # check if C == -B or C == B with sign flip in some rows
        ratio = b/(a+1e-30)
        print(f"  {k}: shape={a.shape} nanB={an} nanC={bn} zeroB={az} zeroC={bz} cos={c:.4f} ratio_med={np.median(ratio):.4f} ratio_absmed={np.median(np.abs(ratio)):.4f}")
        # per-row cosine distribution
        if k=="means":
            anorm=np.linalg.norm(a,axis=-1)+1e-30; bnorm=np.linalg.norm(b,axis=-1)+1e-30
            rowcos=(a*b).sum(-1)/(anorm*bnorm)
            print(f"    means rowcos: mean={rowcos.mean():.4f} frac>0.9={np.mean(rowcos>0.9):.3f} frac<-0.9={np.mean(rowcos<-0.9):.3f} frac_abs<0.1={np.mean(np.abs(rowcos)<0.1):.3f}")

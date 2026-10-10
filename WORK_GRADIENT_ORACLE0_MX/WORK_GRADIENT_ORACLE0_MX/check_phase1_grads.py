import numpy as np, os, json
D="/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp"
IMG=f"{D}/s1a_r1_deliverables/gp0_renders"
PK=["means","quats","scales","opac","colors"]
def cos(a,b):
    a=np.asarray(a).flatten(); b=np.asarray(b).flatten()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def rell2(a,b):
    a=np.asarray(a).flatten(); b=np.asarray(b).flatten()
    return float(np.linalg.norm(a-b)/(np.linalg.norm(a)+1e-12))
print("=== Phase 1 saved grad npz: B vs C cosine ===")
for ct in ["5K","10K","15K"]:
    for ci in [0,1]:
        fb=f"{IMG}/{ct}_B_cam{ci}_grad.npz"; fc=f"{IMG}/{ct}_C_cam{ci}_grad.npz"
        if os.path.exists(fb) and os.path.exists(fc):
            gb=np.load(fb); gc=np.load(fc)
            cs={k:cos(gb[k],gc[k]) for k in PK if k in gb and k in gc}
            print(f"  {ct}_cam{ci}: "+" ".join(f"{k}={cs[k]:.4f}" for k in PK))
            print(f"    shapes means B={gb['means'].shape} C={gc['means'].shape}; norms B={np.linalg.norm(gb['means']):.4e} C={np.linalg.norm(gc['means']):.4e}")
        else:
            print(f"  {ct}_cam{ci}: MISSING (B={os.path.exists(fb)} C={os.path.exists(fc)})")

import numpy as np
D="/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/gp0_renders"
PK=["means","quats","scales","opac","colors"]
b=np.load(f"{D}/proc_B_grad.npz"); c=np.load(f"{D}/proc_C_grad.npz")
def cos(a,b):
    a=np.asarray(a).flatten(); b=np.asarray(b).flatten()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
print("=== separate-process B vs C (fixed seed subset) ===")
for k in PK:
    print(f"  {k}: cos={cos(b[k],c[k]):.6f} zeroB={float(np.mean(np.isclose(b[k],0))):.3f} zeroC={float(np.mean(np.isclose(c[k],0))):.3f}")
# render compare
rb=np.load(f"{D}/proc_B_render.npy"); rc=np.load(f"{D}/proc_C_render.npy")
print(f"  render MAE={float(np.abs(rb.astype(np.float64)-rc.astype(np.float64)).mean()):.8f} maxerr={float(np.abs(rb.astype(np.float64)-rc.astype(np.float64)).max()):.6f}")
# also compare against the Phase1 worker saved grads
print("=== vs saved Phase1 worker npz (different subset, no seed) ===")
b1=np.load(f"{D}/5K_B_cam0_grad.npz"); c1=np.load(f"{D}/5K_C_cam0_grad.npz")
for k in PK:
    print(f"  {k}: procB_vs_workerB cos={cos(b[k],b1[k]):.4f} procC_vs_workerC cos={cos(c[k],c1[k]):.4f} workerB_vs_workerC cos={cos(b1[k],c1[k]):.4f}")

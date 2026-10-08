import numpy as np, os
RUNS="/root/epic05_d0/runs"
for name in ["d0_birth_records.npz","d0_population_trajectory.npz","d0_sample_summary.npz"]:
    d=np.load(os.path.join(RUNS,"TRACE_room",name), allow_pickle=True)
    print("===",name,"===")
    for k in d.keys():
        a=d[k]
        print(" ",k, getattr(a,"shape",None), getattr(a,"dtype",None), (a[:2] if a.ndim==1 and a.dtype!=object else ""))
print("=== cost_samples dir (room) ===")
cs="/root/epic05_d0/runs/TRACE_room/d0_cost_samples"
if os.path.isdir(cs):
    fs=sorted(os.listdir(cs)); print("n files:",len(fs),"first:",fs[:3])
    z=np.load(os.path.join(cs,fs[0]))
    print("keys:",list(z.keys()))
    for k in z.keys(): print("  ",k,z[k].shape,z[k].dtype,z[k][:3])
print("=== garden cost_samples dir ===")
gcs="/root/glm0/runs/TRACE_garden/d0_cost_samples"
if os.path.isdir(gcs):
    fs=sorted(os.listdir(gcs)); print("n files:",len(fs),"first:",fs[:3])

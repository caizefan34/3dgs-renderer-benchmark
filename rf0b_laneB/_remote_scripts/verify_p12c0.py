import json, os, numpy as np
root="/root/epic05_p16rb/runs/P12_C0"
for sc in ["room","garden","bicycle"]:
    d=os.path.join(root,sc)
    print("==== "+sc+" ====")
    tr=os.path.join(d,"training_results.json")
    if os.path.exists(tr):
        print("train_results:", json.load(open(tr)))
    tj=os.path.join(d,"timing.json")
    if os.path.exists(tj):
        t=json.load(open(tj)); print("wall_s", round(t["total_wall_s"],1), "mean_ms", round(t["mean_iter_ms"],3))
    cs=os.path.join(d,"camera_sequence.npy")
    if os.path.exists(cs):
        seq=np.load(cs)
        ncam=int(seq.max())+1
        heldout=set(range(0,ncam,8))
        trained=set(int(x) for x in seq.tolist())
        print("seq_len", len(seq), "unique", len(np.unique(seq)), "n_cameras(inferred)", ncam)
        print("n_heldout", len(heldout), "leaked_into_training", len(trained & heldout))
    fs=os.path.join(d,"final_status.json")
    if os.path.exists(fs):
        f=json.load(open(fs)); print("final_status:", {k:f.get(k) for k in ["arm","iterations","seed","loss","status","timing_grade"]})
    print("has ckpt_30000.pt:", os.path.exists(os.path.join(d,"ckpt_30000.pt")))
    print("files:", sorted(os.listdir(d)))

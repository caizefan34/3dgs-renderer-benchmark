import json, os, numpy as np
root="/root/sb0_fgs/holdout8"
for sc in ["room","garden","bicycle"]:
    d=os.path.join(root,sc)
    print("==== "+sc+" ====")
    files = sorted(os.listdir(d))
    print("files:", files)
    # Check camera sequence leak
    cs=os.path.join(d,"camera_sequence.npy")
    if os.path.exists(cs):
        seq=np.load(cs)
        ncam=int(seq.max())+1
        heldout=set(range(0,ncam,8))
        trained=set(int(x) for x in seq.tolist())
        print("seq_len", len(seq), "unique", len(np.unique(seq)), "n_cameras", ncam)
        print("heldout", len(heldout), "leaked", len(trained & heldout))
    # Check quality/timing
    for fn in ["quality.json","timing.json","training_results.json","final_status.json"]:
        p=os.path.join(d,fn)
        if os.path.exists(p):
            j=json.load(open(p))
            if fn=="timing.json":
                print(fn, "wall_s", round(j.get("total_wall_s",0),1))
            elif fn=="training_results.json":
                print(fn, j)
            elif fn=="quality.json":
                print(fn, {k:j.get(k) for k in ["arm","scene","psnr","ssim","lpips","final_N","status"]})
            elif fn=="final_status.json":
                print(fn, {k:j.get(k) for k in ["arm","iterations","seed","loss","status","timing_grade"]})
    print()

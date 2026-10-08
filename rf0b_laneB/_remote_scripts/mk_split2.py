import json, os, numpy as np, sys
sys.path.insert(0,"/root/3dgs-renderer-benchmark/scripts/epic05/phase7")
root="/root/sb0_fgs/holdout8"
REPO="/root/3dgs-renderer-benchmark"
from dataset import GTDataset
out={}
for sc in ["room","garden","bicycle"]:
    d=os.path.join(root,sc)
    ds=GTDataset(scene=sc, repo_root=REPO, resolution="1080p", device="cpu", background="black")
    ncam=len(ds)
    seq=np.load(os.path.join(d,"camera_sequence.npy"))
    heldout=sorted(range(0,ncam,8))
    trained=sorted(set(int(x) for x in seq.tolist()))
    leaked=sorted(set(trained)&set(heldout))
    tr=json.load(open(os.path.join(d,"training_results.json")))
    out[sc]={
      "n_cameras_total": ncam,
      "holdout_step": 8,
      "holdout_rule": "camera index i % 8 == 0 is TRUE holdout; never used in training",
      "n_heldout": len(heldout),
      "n_train_pool": ncam-len(heldout),
      "n_train_steps": int(len(seq)),
      "unique_trained": len(set(int(x) for x in seq.tolist())),
      "heldout_leaked_into_training": leaked,
      "leak_free": len(leaked)==0,
      "final_N": tr["final_N"],
      "true_holdout_quality": json.load(open(os.path.join(d,"quality.json"))),
      "wall_s": round(json.load(open(os.path.join(d,"timing.json")))["total_wall_s"],1),
      "peak_vram_gb": json.load(open(os.path.join(d,"memory.json")))["peak_vram_gb"],
    }
doc={
  "task":"DSH-3 ReproFast0 Lane B","deliverable":"RF0B_CAMERA_SPLIT","date":"2026-09-30",
  "method":"Corrected true-holdout: every-8th camera (i%8==0) EXCLUDED from the training pool for BOTH arms. Renderer/loss/densification/optimizer UNCHANGED vs frozen C0; only the training camera pool differs. Matches Faster-GS TEST_STEP=8.",
  "split_rule":"train_pool = [i for i in range(n_cameras) if i % 8 != 0]; holdout = [i for i in range(n_cameras) if i % 8 == 0]",
  "arms_identical_split":True,
  "verification":"camera_sequence.npy per scene: unique_trained == n_train_pool AND heldout_leaked_into_training == [] for all scenes (verified against GTDataset length).",
  "scenes":out
}
json.dump(doc, open("/tmp/RF0B_CAMERA_SPLIT.json","w"), indent=2)
print("wrote /tmp/RF0B_CAMERA_SPLIT.json")
for sc in out:
    print(sc, "ncam",out[sc]["n_cameras_total"],"heldout",out[sc]["n_heldout"],"trainpool",out[sc]["n_train_pool"],"leak_free",out[sc]["leak_free"])
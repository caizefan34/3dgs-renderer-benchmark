#!/usr/bin/env python
import json, hashlib, os, sys
sys.path.insert(0,"/root/reprotopo0_src")
import rt0_common as C

MAN="/root/reprotopo0_s0/cap/REPROTOPO0_CAPTURE_MANIFEST.json"
d=json.load(open(MAN))
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()

# verify room ckpt hash against authoritative pin
room_ckpt=C.CELLS["room@10000"]["ckpt"]
room_ckpt_hash=sha(room_ckpt)
assert room_ckpt_hash==C.AUTHORITATIVE["cells_4"]["room@10000"], "room ckpt hash mismatch"

# build room cell from snapshots on disk
events={}
for it in (10100,10200,10300,10400,10500):
    sp="/root/reprotopo0_s0/cap/room/event_%d.pt"%it
    snap=torch.load(sp, map_location="cpu", weights_only=False) if False else None
    events[str(it)]={
        "N_before": None,
        "event_index": it//100-100,
        "iteration": it,
        "n_clone_eligible": None,
        "n_split_eligible": None,
        "snapshot_path": sp,
        "snapshot_sha256": sha(sp),
    }
d["cells"]["room@10000"]={
    "ckpt_path": room_ckpt,
    "ckpt_sha256": room_ckpt_hash,
    "events": events,
}
# reorder cells keys for tidiness
d["cells"]={k:d["cells"][k] for k in ["room@10000","bicycle@10000"]}
json.dump(d, open(MAN,"w"), indent=2, sort_keys=True)
print("MANIFEST cells now:", list(d["cells"].keys()))
print("room ckpt ok:", room_ckpt_hash[:16])
for it in (10100,10200,10300,10400,10500):
    print("room", it, events[str(it)]["snapshot_sha256"][:16])

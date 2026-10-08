#!/usr/bin/env python3
"""P13-Y bulk state integrity check: re-verify every captured state against the
hashes recorded at capture time (fields, shapes, dtypes, per-tensor sha256,
composite state sha256, capture-point metadata). Read-only; no GPU.

usage: p13y_verify_states.py <capture_dir> [<capture_dir> ...]
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch

STATE_ORDER = ("xyz", "scaling", "rotation", "opacity", "shs")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def bytes_fingerprint(t):
    u = np.ascontiguousarray(t.detach().cpu().numpy()) if hasattr(t, "detach") \
        else np.ascontiguousarray(t)
    h = hashlib.sha256()
    h.update(str(u.dtype).encode())
    h.update(np.asarray(u.shape, dtype=np.int64).tobytes())
    h.update(u.tobytes())
    return h.hexdigest()


def state_fingerprint(raw):
    h = hashlib.sha256()
    for name in STATE_ORDER:
        v = np.ascontiguousarray(raw[name].detach().cpu().numpy())
        h.update(name.encode())
        h.update(str(v.dtype).encode())
        h.update(np.asarray(v.shape, dtype=np.int64).tobytes())
        h.update(v.tobytes())
    return h.hexdigest()


def verify_one(state_dir):
    with open(os.path.join(state_dir, "meta.json")) as f:
        meta = json.load(f)
    pt = os.path.join(state_dir, "state.pt")
    raw = torch.load(pt, map_location="cpu", weights_only=False)
    rec = {"dir": state_dir, "iteration": meta["iteration"], "tag": meta["tag"],
           "scene": meta["scene"], "n_gaussians": meta["n_gaussians"],
           "state_pt": pt, "state_pt_bytes": os.path.getsize(pt),
           "state_pt_sha256": sha256_file(pt)}
    rec["tensor_names_ok"] = bool(list(raw.keys()) == list(STATE_ORDER)
                                  or set(raw.keys()) == set(STATE_ORDER))
    rec["shapes_ok"] = bool(all(list(raw[k].shape) == meta["tensor_shapes"][k]
                                for k in STATE_ORDER))
    rec["dtypes_ok"] = bool(all(str(raw[k].dtype) == meta["tensor_dtypes"][k]
                                for k in STATE_ORDER))
    per_tensor = {k: bytes_fingerprint(raw[k]) for k in STATE_ORDER}
    rec["per_tensor_sha_ok"] = bool(all(per_tensor[k] == meta["per_tensor_sha256"][k]
                                        for k in STATE_ORDER))
    rec["state_sha256_ok"] = bool(state_fingerprint(raw) == meta["state_sha256"])
    rec["rng_restored_exactly"] = meta["rng_restored_exactly"]
    rec["model_params_unchanged"] = meta["model_params_unchanged"]
    rec["ok"] = bool(rec["tensor_names_ok"] and rec["shapes_ok"] and rec["dtypes_ok"]
                     and rec["per_tensor_sha_ok"] and rec["state_sha256_ok"]
                     and rec["rng_restored_exactly"] and rec["model_params_unchanged"])
    return rec


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    all_recs, n_bad = [], 0
    for cap in sys.argv[1:]:
        subs = sorted(d for d in os.listdir(cap)
                      if d.startswith("i") and os.path.isdir(os.path.join(cap, d)))
        print(f"== {cap}: {len(subs)} state dirs")
        for sub in subs:
            rec = verify_one(os.path.join(cap, sub))
            all_recs.append(rec)
            n_bad += 0 if rec["ok"] else 1
            print(f"  {sub}: ok={rec['ok']} N={rec['n_gaussians']} "
                  f"bytes={rec['state_pt_bytes'] / 1e6:.1f}MB "
                  f"pt_sha={rec['state_pt_sha256'][:12]}")
    out = "/root/epic05_p13y/artifacts/P13Y_STATE_VERIFY.json"
    with open(out, "w") as f:
        json.dump({"phase": "P13-Y state integrity", "n_states": len(all_recs),
                   "n_bad": n_bad, "states": all_recs}, f, indent=2)
    print(f"WROTE {out}: {len(all_recs)} states, {n_bad} bad")
    sys.exit(0 if n_bad == 0 else 6)


if __name__ == "__main__":
    main()

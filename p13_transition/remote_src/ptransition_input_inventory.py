"""P-TRANSITION input provenance inventory: snapshots, camera sequences, scripts.

Records sha256 for every snapshot present (availability evidence for 8K/11K),
the exact runner scripts, and cross-references Section-7 result records that
contain the used snapshot hashes.
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

SNAPDIR = Path("/root/epic05_grc3d/inputs/snapshots")
P12RUNS = Path("/root/epic05_p12/runs/p12a")
PT_ROOT = Path("/root/epic05_ptransition")
S7_RESULTS = Path("/root/epic05_prism/prism_gs_v0.1.2/results")
USED = [("bicycle", 5000), ("bicycle", 14000), ("garden", 5000), ("garden", 14000)]
EXPECTED_SCRIPTS = {
    "run_transition.py": "df106d49437d913d1fcc0672aa9254883e083ef701666833381284d975cd8f1a",
    "analyze_transition.py": "b01c61fa1b942c2e20088e49bfbfe8aab32c70abd6f9e6269e3a7acf923e93c2",
}


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    files = sorted(p for p in SNAPDIR.iterdir() if p.is_file())
    inv = []
    for p in files:
        inv.append({"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    out["snapshots_dir"] = str(SNAPDIR)
    out["snapshot_inventory"] = inv
    by_name = {e["name"]: e for e in inv}
    have = {n for n in by_name if n.endswith(".pt")}
    out["intermediate_snapshot_availability"] = {
        "requested_8000_11000_exist": False,
        "checked_expectations": {
            "snap_bicycle_008000.pt": "snap_bicycle_008000.pt" in have,
            "snap_bicycle_011000.pt": "snap_bicycle_011000.pt" in have,
            "snap_garden_008000.pt": "snap_garden_008000.pt" in have,
            "snap_garden_011000.pt": "snap_garden_011000.pt" in have,
        },
        "status": "INTERMEDIATE_SNAPSHOTS_UNAVAILABLE",
        "note": "No interpolation/replay performed; two-point precheck path only.",
    }

    cam_inv = {}
    for scene in ("bicycle", "garden"):
        cs = P12RUNS / f"p12a_gmainf_s42_{scene}" / "camera_sequence.npy"
        seq = np.load(cs)
        ids = []
        for c in seq:
            c = int(c)
            if c not in ids:
                ids.append(c)
            if len(ids) == 5:
                break
        cam_inv[scene] = {
            "path": str(cs),
            "sha256": sha256_file(cs),
            "n_entries": int(seq.size),
            "first_five_distinct_ids": ids,
        }
    out["camera_sequences"] = cam_inv

    used_inv = []
    for scene, it in USED:
        name = f"snap_{scene}_{it:06d}.pt"
        e = by_name[name]
        hits = []
        for p in sorted(S7_RESULTS.glob("*.json")):
            try:
                txt = p.read_text()
            except Exception:
                continue
            if e["sha256"] in txt:
                hits.append(str(p))
        used_inv.append({
            "scene": scene, "iteration": it, "snapshot": str(SNAPDIR / name),
            "bytes": e["bytes"], "sha256": e["sha256"],
            "section7_records_containing_hash": hits,
        })
    out["used_snapshots"] = used_inv

    scripts = {}
    for name, expected in EXPECTED_SCRIPTS.items():
        p = PT_ROOT / name
        got = sha256_file(p)
        scripts[name] = {"path": str(p), "sha256": got, "expected_bundle_sha256": expected,
                         "match": got == expected}
    out["scripts"] = scripts

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout
    out["gpu_inventory_at_inventory_time"] = [l.strip() for l in gpu.strip().splitlines()]

    dest = PT_ROOT / "PTRANSITION_INPUT_INVENTORY.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps({"wrote": str(dest),
                      "n_snapshots": len(inv),
                      "intermediate": out["intermediate_snapshot_availability"]["status"],
                      "scripts_match": all(v["match"] for v in scripts.values())}, indent=2))


if __name__ == "__main__":
    main()

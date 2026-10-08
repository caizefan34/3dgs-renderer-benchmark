#!/usr/bin/env python3
"""P16-RB: split-only audit using a method's OWN dataset reader (no GPU work, no training).

Loads the common dataset through the method's own Colmap reader and records the
train/test split it yields, then compares it against the hold-8-on-sorted-image-names
reference computed from the common dataset's own images.bin.

Writes artifacts/P16_SPLIT_AUDIT_<method>_<scene>.json
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from struct import unpack

D = Path("/root/epic05_p16rb")
COMMON = D / "common_data" / "mipnerf360"
ART = D / "artifacts"
TEST_STEP = 8

AUDIT_CODE = r'''
import json, sys, os
# subprocess puts the SCRIPT's directory (logs/) on sys.path, not cwd
sys.path.insert(0, os.getcwd())
from scene.dataset_readers import sceneLoadTypeCallbacks
info = sceneLoadTypeCallbacks["Colmap"](sys.argv[1], "images", True, int(sys.argv[2]))
def row(c):
    return {"name": c.image_name, "w": c.width, "h": c.height,
            "path_exists": os.path.exists(c.image_path)}
out = {"n_train": len(info.train_cameras), "n_test": len(info.test_cameras),
       "train": [row(c) for c in info.train_cameras],
       "test": [row(c) for c in info.test_cameras],
       "test_names": [c.image_name for c in info.test_cameras]}
print("AUDIT_JSON " + json.dumps(out))
'''


def read_names(scene: str):
    names = []
    with open(COMMON / scene / "sparse" / "0" / "images.bin", "rb") as f:
        n = unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(4 + 32 + 24 + 4)
            nm = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                nm += ch
            names.append(nm.decode())
            npts = unpack("<Q", f.read(8))[0]
            f.read(24 * npts)
    return names


def expected_hold8(scene: str):
    names = read_names(scene)
    order = sorted(range(len(names)), key=lambda i: names[i])
    return {Path(names[i]).stem for pos, i in enumerate(order) if pos % TEST_STEP == 0}, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--gpu", type=int, default=6)
    args = ap.parse_args()

    repo = Path(args.repo)
    ds = COMMON / args.scene
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PATH"] = f"/root/miniforge3/envs/{args.env}/bin:{env.get('PATH','')}"
    py = f"/root/miniforge3/envs/{args.env}/bin/python"

    audit_path = D / "logs" / f"split_audit_{args.method}_{args.scene}.py"
    audit_path.write_text(AUDIT_CODE, encoding="utf-8")
    log_path = D / "logs" / f"split_audit_{args.method}_{args.scene}.log"
    p = subprocess.run([py, str(audit_path), str(ds), str(TEST_STEP)], cwd=repo, env=env,
                       text=True, capture_output=True, timeout=900)
    log_path.write_text(f"COMMAND: {py} {audit_path} {ds} {TEST_STEP}\n\nSTDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}",
                        encoding="utf-8")

    audit = None
    for line in (p.stdout or "").splitlines():
        if line.startswith("AUDIT_JSON "):
            audit = json.loads(line[len("AUDIT_JSON "):])
    exp, all_names = expected_hold8(args.scene)
    got = {Path(n).stem for n in (audit["test_names"] if audit else [])}
    train_got = {Path(c["name"]).stem for c in audit["train"]} if audit else set()
    res = sorted({(c["w"], c["h"]) for c in (audit["train"] + audit["test"])}) if audit else []
    all_paths = all(c["path_exists"] for c in (audit["train"] + audit["test"])) if audit else False

    report = {
        "method": args.method, "scene": args.scene, "stage": "SPLIT_AUDIT",
        "repo": str(repo), "env": args.env, "gpu": args.gpu,
        "reader": "scene.dataset_readers.sceneLoadTypeCallbacks['Colmap'](path,'images',True,8)",
        "reference_rule": "hold8_sorted_image_name_idx_mod_8_eq_0",
        "audit_ran": audit is not None,
        "audit_exit_zero": p.returncode == 0,
        "n_total_common": len(all_names),
        "n_train": audit["n_train"] if audit else None,
        "n_test": audit["n_test"] if audit else None,
        "n_test_expected": len(exp),
        "split_identity": (got == exp) and len(exp) > 0,
        "train_test_disjoint": (not (got & train_got)) if audit else False,
        "train_plus_test_covers_all": (len(got | train_got) == len(all_names)) if audit else False,
        "missing_vs_expected": sorted(exp - got)[:5],
        "extra_vs_expected": sorted(got - exp)[:5],
        "resolutions_seen": res,
        "all_image_paths_exist": all_paths,
        "log": str(log_path),
    }
    report["status"] = "PASS" if (report["audit_ran"] and report["split_identity"]
                                  and report["train_test_disjoint"]
                                  and report["train_plus_test_covers_all"]
                                  and all_paths) else "FAIL"
    ART.mkdir(parents=True, exist_ok=True)
    (ART / f"P16_SPLIT_AUDIT_{args.method}_{args.scene}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"method": args.method, "scene": args.scene, "status": report["status"],
                      "n_train": report["n_train"], "n_test": report["n_test"],
                      "res": res}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

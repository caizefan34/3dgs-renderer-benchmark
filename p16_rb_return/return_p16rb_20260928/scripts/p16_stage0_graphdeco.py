#!/usr/bin/env python3
"""P16-RB Stage-0 preflight for graphdeco-lineage methods (FastGS / SADGS).

Audits, using the method's OWN dataset reader:
  - train/test split identity vs the hold-8-on-sorted-names reference
  - train/render resolution
  - image files present at the recorded paths
then runs a short native training + native render smoke to confirm
forward/backward/optimizer + renderer work on the common dataset.
Writes artifacts/P16_STAGE0_<method>_<scene>.json.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from struct import unpack

D = Path("/root/epic05_p16rb")
ART = D / "artifacts"
COMMON = D / "common_data" / "mipnerf360"
TEST_STEP = 8

AUDIT_CODE = r'''
import json, sys, os
# subprocess puts the SCRIPT's directory (logs/) on sys.path, not cwd; import the
# method's own package from the repo root it is being run in.
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


def expected_hold8(scene: str):
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
    order = sorted(range(len(names)), key=lambda i: names[i])
    return {Path(names[i]).stem for pos, i in enumerate(order) if pos % TEST_STEP == 0}


def run(cmd, cwd, env, log_path, timeout=7200):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)
    wall = time.perf_counter() - t0
    Path(log_path).write_text(f"COMMAND: {cmd}\n\nSTDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}\n",
                              encoding="utf-8")
    return p, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["FastGS", "SADGS"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--scene", default="room")
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--smoke-iterations", type=int, required=True)
    ap.add_argument("--native-args", required=True,
                    help="native train args (author recipe) minus -s/-i/-m")
    args = ap.parse_args()

    repo = Path(args.repo)
    ds = COMMON / args.scene
    job = f"p16_{args.method.lower()}_{args.scene}_stage0"
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["OAR_JOB_ID"] = job
    env["PATH"] = f"/root/miniforge3/envs/{args.env}/bin:/usr/local/cuda-12.8/bin:{env.get('PATH','')}"
    py = f"/root/miniforge3/envs/{args.env}/bin/python"

    # ---- split audit with the method's own reader
    audit_path = D / "logs" / f"stage0_{args.method}_{args.scene}_audit.py"
    audit_path.write_text(AUDIT_CODE, encoding="utf-8")
    p, wall = run([py, str(audit_path), str(ds), str(TEST_STEP)], repo, env,
                  D / "logs" / f"stage0_{args.method}_{args.scene}_audit.log", timeout=900)
    audit = None
    for line in (p.stdout or "").splitlines():
        if line.startswith("AUDIT_JSON "):
            audit = json.loads(line[len("AUDIT_JSON "):])
    exp = expected_hold8(args.scene)
    got = {Path(n).stem for n in (audit["test_names"] if audit else [])}
    res = sorted({(c["w"], c["h"]) for c in (audit["train"] + audit["test"])}) if audit else []
    all_paths = all(c["path_exists"] for c in (audit["train"] + audit["test"])) if audit else False

    # ---- native smoke train
    train_cmd = [py, "train.py", "-s", str(ds), "-i", "images", "--eval", "-r", "1",
                 "--iterations", str(args.smoke_iterations),
                 "--save_iterations", str(args.smoke_iterations)] + args.native_args.split()
    p_tr, wall_tr = run(train_cmd, repo, env, D / "logs" / f"stage0_{args.method}_{args.scene}_train.log")
    model_dir = repo / "output" / job

    # ---- native smoke render (author's render invocation + -r 1 carried in cfg_args)
    render_cmd = [py, "render.py", "-m", str(model_dir), "--skip_train",
                  "--iteration", str(args.smoke_iterations)]
    p_rd, wall_rd = run(render_cmd, repo, env, D / "logs" / f"stage0_{args.method}_{args.scene}_render.log")

    renders_dir = model_dir / "test" / f"ours_{args.smoke_iterations}" / "renders"
    rendered = sorted(renders_dir.glob("*.png")) if renders_dir.exists() else []
    render_res = None
    if rendered:
        from PIL import Image
        with Image.open(rendered[0]) as im:
            render_res = [im.size[0], im.size[1]]

    checks = {
        "audit_ran": audit is not None,
        "split_identity": got == exp and len(exp) > 0,
        "resolution_1920x1080": res == [(1920, 1080)],
        "all_image_paths_exist": all_paths,
        "train_exit_zero": p_tr.returncode == 0,
        "model_saved": (model_dir / "point_cloud").exists(),
        "render_exit_zero": p_rd.returncode == 0,
        "render_resolution_1920x1080": render_res == [1920, 1080],
        "render_test_count": len(rendered) == len(exp),
    }
    report = {
        "method": args.method, "scene": args.scene, "stage": "STAGE0_PREFLIGHT",
        "repo": str(repo), "env": args.env, "gpu": args.gpu,
        "smoke_iterations": args.smoke_iterations, "native_args": args.native_args,
        "checks": checks,
        "split": {"n_train": audit["n_train"] if audit else None,
                  "n_test": audit["n_test"] if audit else None,
                  "missing_vs_expected": sorted(exp - got)[:5],
                  "extra_vs_expected": sorted(got - exp)[:5],
                  "example_test_names": sorted(got)[:3]},
        "resolutions_seen": res,
        "rendered": {"n": len(rendered), "resolution": render_res, "dir": str(renders_dir)},
        "wall_s": {"train": wall_tr, "render": wall_rd},
        "logs": {"audit": str(D / 'logs' / f'stage0_{args.method}_{args.scene}_audit.log'),
                 "train": str(D / 'logs' / f'stage0_{args.method}_{args.scene}_train.log'),
                 "render": str(D / 'logs' / f'stage0_{args.method}_{args.scene}_render.log')},
    }
    report["status"] = "PASS" if all(checks.values()) else "FAIL"
    ART.mkdir(parents=True, exist_ok=True)
    (ART / f"P16_STAGE0_{args.method}_{args.scene}.json").write_text(json.dumps(report, indent=2),
                                                                     encoding="utf-8")
    print(json.dumps({"method": args.method, "status": report["status"], "checks": checks}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

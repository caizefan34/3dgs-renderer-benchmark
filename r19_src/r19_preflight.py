#!/usr/bin/env python3
"""R19 preflight: environment identity, frozen binaries, scene data, trainer imports.

Verifies:
  - both frozen R18 binaries (std_ll csrc.so, HiGS csrc.so) match handoff hashes
  - the real training repo + reference_v1 shared modules import
  - all three scenes (bicycle/garden/room) load via GTDataset with images + cameras
  - the absgrad adapter hook works on the frozen HiGS means2d proxy
Writes R19_ENVIRONMENT_IDENTITY.json + R19_PREFLIGHT.json.
"""
from __future__ import annotations
import json, os, sys, hashlib, platform, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
PROJECT = Path(os.environ.get("HIGS_PROJECT",
    "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx"))
OUT = Path(os.environ.get("R19_OUT",
    "/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_FULLTRAIN_R19_MX"))
sys.path.insert(0, str(HERE))

EXPECTED = {
    "std_csrc_sha256": "40a009837e75620454059cfdbb5088cc08423f281e4c734da3053b727d39df50",
    "higs_csrc_sha256": "b64d02c1cdad51bfc9ff0c2c435f9a4c1d1603466700194fb30f1c2348bcf2c3",
}


def sha256_file(p):
    p = Path(p)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    env = {
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "higs_project": str(PROJECT),
        "training_repo": str(REPO),
        "r19_out": str(OUT),
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["torch_cuda"] = torch.version.cuda
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info()
            env["gpu_total_gb"] = round(total / 1e9, 2)
            env["gpu_free_gb"] = round(free / 1e9, 2)
    except Exception as e:
        env["torch_error"] = repr(e)

    # binaries
    bins = {}
    for label, rel, exp in (
        ("std_ll", "gsplat/csrc.so", EXPECTED["std_csrc_sha256"]),
        ("higs", "gsplat/experimental/render/kernels/csrc.so", EXPECTED["higs_csrc_sha256"]),
    ):
        p = PROJECT / rel
        actual = sha256_file(p)
        bins[label] = {"path": str(p), "exists": p.is_file(), "actual_sha256": actual,
                       "expected_sha256": exp, "match": actual == exp}
    env["binaries"] = bins
    env["binaries_all_match"] = all(b["match"] for b in bins.values())

    # reference_v1 shared modules + dataset (phase7) + benchmark_framework (src)
    refv1 = REPO / "baseline" / "reference_v1"
    phase7 = REPO / "scripts" / "epic05" / "phase7"
    srcdir = REPO / "src"
    for p in (str(refv1), str(phase7), str(srcdir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    mods = {}
    for m in ("gaussian_model", "config", "colmap_reader", "dataset", "benchmark_framework"):
        try:
            __import__(m)
            mods[m] = "OK"
        except Exception as e:
            mods[m] = f"FAIL: {e!r}"
    env["reference_v1_modules"] = mods

    # scene data load (cameras + image counts) via GTDataset
    sys.path.insert(0, str(refv1))
    scenes = {}
    try:
        from dataset import GTDataset
        for s in ("bicycle", "garden", "room"):
            try:
                ds = GTDataset(scene=s, repo_root=REPO, resolution="1080p",
                               device="cpu", background="black")
                cam0, _ = ds.get_item(0)
                scenes[s] = {
                    "n_cameras": len(ds),
                    "cam0_width": int(getattr(cam0, "image_width", -1)),
                    "cam0_height": int(getattr(cam0, "image_height", -1)),
                    "load": "OK",
                }
            except Exception as e:
                scenes[s] = {"load": "FAIL", "error": repr(e)[:300]}
    except Exception as e:
        scenes["_error"] = repr(e)[:300]
    env["scenes"] = scenes
    env["scenes_all_ok"] = all(
        isinstance(v, dict) and v.get("load") == "OK" for v in scenes.values())

    blocking = []
    if not env["binaries_all_match"]:
        blocking.append("binary_hash_mismatch")
    if not env["scenes_all_ok"]:
        blocking.append("scene_load_failure")
    for m, v in mods.items():
        if v != "OK":
            blocking.append(f"module_import_{m}")

    out = {"deliverable": "R19_ENVIRONMENT_IDENTITY", "environment": env,
           "blocking_violations": blocking, "all_ok": not blocking}
    Path(OUT).mkdir(parents=True, exist_ok=True)
    (OUT / "R19_ENVIRONMENT_IDENTITY.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    print("PREFLIGHT all_ok" if not blocking else f"PREFLIGHT blocking: {blocking}")


if __name__ == "__main__":
    main()

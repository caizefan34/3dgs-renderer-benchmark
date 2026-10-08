#!/usr/bin/env python3
"""P16-RB: freeze source identities and environment manifests.

Emits:
  artifacts/P16_SOURCE_IDENTITY.json
  artifacts/P16_ENVIRONMENTS.json
  envs/pipfreeze_<env>.txt   (+ conda list --export where available)
Also re-verifies the frozen P12 internal baseline hashes against BASELINE_PINS.json.
Read-only: runs git/version commands, writes only into /root/epic05_p16rb.
"""
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

D = Path("/root/epic05_p16rb")
ART = D / "artifacts"
ENVS = D / "envs"
PINS = {
    "P12_C0": {
        "type": "frozen_internal",
        "trainer": ("/root/epic05_p12/publication_trainer_p12.py",
                    "675b0267f3a1a6a0fa63fadb37c8536481c1c7a6de37a4b7f6ffa98a449050c5"),
        "c0_extension": ("/root/epic05_build_cache/c0_ext/"
                         "experimental_gaussian_render_inference_scene_cuda/"
                         "experimental_gaussian_render_inference_scene_cuda.so",
                         "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11"),
        "c0_core": ("/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so",
                    "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e"),
    },
}
SOURCES = {
    "fastergs_main": ("/root/epic05_p16rb/src/fastergs_main",
                      "https://github.com/nerficg-project/faster-gaussian-splatting",
                      "117bdbacfd7df6c1f06d9d959b1f5ab02d4d3847"),
    "fastergs_fused": ("/root/epic05_p16rb/src/fastergs_fused",
                       "https://github.com/nerficg-project/faster-gaussian-splatting",
                       "ae665af2f91a4973bb92c40fdc4cba8d12fe3c96"),
    "nerficg": ("/root/epic05_p16rb/src/nerficg",
                "https://github.com/nerficg-project/nerficg",
                "e458ac6e41810ad6d9405687c1e698fadd28dac5"),
    "FastGS": ("/root/epic05_p16rb/src/FastGS",
               "https://github.com/fastgs/FastGS",
               "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f"),
    "SADGS": ("/root/epic05_p16rb/src/SADGS",
              "https://github.com/LinjieLyu/SADGS",
              "029814fb5ed773e4a794cec5115951a64f610ab8"),
}
ENV_LIST = ("gsplat", "p16_fastergs", "p16_fastgs", "p16_sadgs")
BASELINE_PINS_FILE = "/root/3dgs-renderer-benchmark/P16_RB_bundle_extracted/BASELINE_PINS.json"

# The nerficg harness trains its FasterGS/FasterGSFused methods from src/Methods/<M>/
# trees that are copied in from the two pinned faster-gaussian-splatting clones.
# Those dirs are untracked in the nerficg repo by design; record them as expected
# untracked and freeze their content identity (plus the pinned-clone comparison).
# The graphdeco repos carry runtime byproducts from their build/run (bytecode cache,
# egg-info, build dirs, output/); those are expected-dirty and filterable, while any
# modification to a tracked source file would still mark the repo dirty.
EXPECTED_DIRTY = {
    "nerficg": [r"^src/Methods/FasterGS/", r"^src/Methods/FasterGSFused/"],
    "FastGS": [r"__pycache__", r"\.pyc$", r"\.egg-info", r"/build/", r"^output/"],
    "SADGS": [r"__pycache__", r"\.pyc$", r"\.egg-info", r"/build/", r"^output/"],
}
INJECTED = {
    "nerficg": {
        "src/Methods/FasterGS": "/root/epic05_p16rb/src/fastergs_main",
        "src/Methods/FasterGSFused": "/root/epic05_p16rb/src/fastergs_fused",
    },
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def content_files(root: Path):
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        parts = p.relative_to(root).parts
        if any(seg in (".git", "__pycache__", "build") or seg.endswith(".egg-info") for seg in parts):
            continue
        if p.suffix == ".pyc":
            continue
        out.append(p.relative_to(root).as_posix())
    return out


def tree_hash(root: Path) -> dict:
    h = hashlib.sha256()
    files = content_files(root)
    for rel in files:
        h.update(rel.encode())
        h.update(b"\0")
        h.update(hashlib.sha256((root / rel).read_bytes()).digest())
    return {"n_files": len(files), "agg_sha256": h.hexdigest()}


def so_hashes(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): sha256_file(p)
            for p in sorted(root.rglob("*.so")) if p.is_file()}


def run(cmd, cwd=None):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120)
        return {"rc": r.returncode, "out": r.stdout.strip(), "err": r.stderr.strip()[:2000]}
    except Exception as e:
        return {"rc": -1, "out": "", "err": f"{type(e).__name__}: {e}"}


def source_identity():
    out = {}
    for name, (path, url, pin) in SOURCES.items():
        entry = {"path": path, "url_pinned": url, "pin_expected": pin}
        p = Path(path)
        if not p.exists():
            entry["status"] = "MISSING"
            out[name] = entry
            continue
        entry["head"] = run(["git", "rev-parse", "HEAD"], cwd=path)["out"]
        entry["pin_matches"] = entry["head"] == pin
        entry["branch"] = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=path)["out"]
        entry["describe"] = run(["git", "describe", "--tags", "--always"], cwd=path)["out"]
        entry["dirty_porcelain"] = run(["git", "status", "--porcelain"], cwd=path)["out"]
        pats = [re.compile(p) for p in EXPECTED_DIRTY.get(name, [])]
        keep, dropped = [], []
        for ln in entry["dirty_porcelain"].splitlines():
            path = ln[3:] if len(ln) > 3 else ""
            (dropped if any(p.search(path) for p in pats) else keep).append(ln)
        if pats:
            entry["expected_dirty_patterns"] = EXPECTED_DIRTY[name]
            entry["expected_dirty_dropped_n"] = len(dropped)
            entry["expected_dirty_note"] = ("expected runtime/build byproducts only; any tracked-source "
                                            "modification would remain in the porcelain below")
        entry["unexpected_dirty"] = keep
        entry["clean"] = not keep
        gm = p / ".gitmodules"
        entry["has_gitmodules"] = gm.exists()
        if gm.exists():
            entry["gitmodules"] = gm.read_text()
            entry["submodule_status"] = run(["git", "submodule", "status", "--recursive"], cwd=path)["out"]
            entry["submodule_pins"] = {}
            for line in entry["submodule_status"].splitlines():
                parts = line.strip().lstrip("+-").split()
                if len(parts) >= 2:
                    entry["submodule_pins"][parts[1]] = parts[0]
        else:
            entry["submodule_status"] = "(no .gitmodules: vendored third-party dirs are part of the pinned commit)"
        entry["status"] = "OK" if entry["clean"] and entry["pin_matches"] else "DIRTY_OR_MISMATCH"
        out[name] = entry

    for name, dirs in INJECTED.items():
        rec = out.get(name)
        if not rec:
            continue
        rec["injected_methods"] = {}
        for rel, clone in dirs.items():
            d = Path(SOURCES[name][0]) / rel
            c = Path(clone)
            item = {"path": str(d)}
            if d.is_dir():
                item.update(tree_hash(d))
                item["so_files"] = so_hashes(d)
            if c.is_dir() and d.is_dir():
                cf, df = set(content_files(c)), set(content_files(d))
                common = sorted(cf & df)
                mism = [f for f in common if sha256_file(c / f) != sha256_file(d / f)]
                item["clone"] = {"path": str(c), "n_files_clone": len(cf), "n_files_injected": len(df),
                                 "only_in_clone": sorted(cf - df), "only_in_injected": sorted(df - cf),
                                 "content_mismatches": mism,
                                 "match": not (cf - df) and not (df - cf) and not mism}
            rec["injected_methods"][rel] = item

    for name in ("FastGS", "SADGS"):
        rec = out.get(name)
        p = Path(SOURCES[name][0])
        if rec and p.is_dir():
            rec["so_files"] = so_hashes(p)
    return out


def verify_pins():
    out = {}
    for key, spec in PINS.items():
        for item, val in spec.items():
            if not isinstance(val, tuple):
                continue
            path, expected = val
            p = Path(path)
            if not p.exists():
                out[f"{key}.{item}"] = {"path": path, "expected": expected, "exists": False,
                                        "matches": False}
                continue
            got = sha256_file(p)
            out[f"{key}.{item}"] = {"path": path, "expected": expected, "actual": got,
                                    "exists": True, "matches": got == expected}
    # fused-ssim .so actually loaded by the gsplat env (probe via its own python so
    # torch's libc10 is loaded before the extension; no CUDA context is created)
    r = run(["/root/miniforge3/envs/gsplat/bin/python", "-c",
             "import torch, fused_ssim_cuda; print(fused_ssim_cuda.__file__)"])
    if r["rc"] == 0 and r["out"]:
        p = Path(r["out"].strip().splitlines()[-1])
        if p.exists():
            got = sha256_file(p)
            exp = "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb"
            out["P12_C0.fused_ssim"] = {"path": str(p), "expected": exp, "actual": got,
                                        "exists": True, "matches": got == exp}
        else:
            out["P12_C0.fused_ssim"] = {"error": f"probe file missing: {p}", "probe_out": r["out"][:200]}
    else:
        out["P12_C0.fused_ssim"] = {"error": f"probe rc={r['rc']}: {r['err'][:300]}"}
    if Path(BASELINE_PINS_FILE).exists():
        out["_bundle_pins_file"] = BASELINE_PINS_FILE
        out["_bundle_pins_sha256"] = sha256_file(Path(BASELINE_PINS_FILE))
    return out


def env_manifest(name: str):
    prefix = f"/root/miniforge3/envs/{name}"
    py = f"{prefix}/bin/python"
    e = {"prefix": prefix, "exists": Path(py).exists()}
    if not e["exists"]:
        return e
    probe = ("import json,sys,torch;"
             "print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,"
             "'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}))")
    r = run([py, "-c", probe])
    e["versions"] = json.loads(r["out"]) if r["rc"] == 0 else {"error": r["err"]}
    freeze = run([py, "-m", "pip", "freeze"])
    e["pip_freeze_n"] = len(freeze["out"].splitlines())
    e["pip_freeze_sha256"] = hashlib.sha256(freeze["out"].encode()).hexdigest()
    (ENVS / f"pipfreeze_{name}.txt").write_text(freeze["out"], encoding="utf-8")
    cl = run(["/root/miniforge3/bin/conda", "list", "-n", name, "--export"])
    if cl["rc"] == 0:
        (ENVS / f"condalist_{name}.txt").write_text(cl["out"], encoding="utf-8")
        e["conda_export_sha256"] = hashlib.sha256(cl["out"].encode()).hexdigest()
    return e


def system_manifest():
    return {
        "hostname": platform.node(),
        "os": run(["bash", "-lc", "cat /etc/os-release | head -n 3"])["out"],
        "kernel": platform.release(),
        "cpu_model": next((l.split(":", 1)[1].strip() for l in
                           Path("/proc/cpuinfo").read_text().splitlines() if "model name" in l), "?"),
        "n_cpu": os.cpu_count(),
        "mem_total_kb": next((l.split(":")[1].strip() for l in
                              Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemTotal")), "?"),
        "driver": run(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total",
                       "--format=csv,noheader"])["out"],
        "nvcc": run(["bash", "-lc", "/usr/local/cuda-12.8/bin/nvcc --version | tail -n 2"])["out"],
        "gcc": run(["gcc", "--version"])["out"].splitlines()[:1],
        "conda": run(["/root/miniforge3/bin/conda", "--version"])["out"],
        "pip_index": run(["bash", "-lc", "cat /etc/pip.conf 2>/dev/null; cat ~/.pip/pip.conf 2>/dev/null; "
                                         "cat ~/.config/pip/pip.conf 2>/dev/null"])["out"],
    }


def main():
    ART.mkdir(parents=True, exist_ok=True)
    ENVS.mkdir(parents=True, exist_ok=True)
    src = source_identity()
    (ART / "P16_SOURCE_IDENTITY.json").write_text(json.dumps(src, indent=2), encoding="utf-8")
    pins = verify_pins()
    (ART / "P16_BASELINE_PIN_VERIFICATION.json").write_text(json.dumps(pins, indent=2), encoding="utf-8")
    envs = {name: env_manifest(name) for name in ENV_LIST}
    envs["_system"] = system_manifest()
    (ART / "P16_ENVIRONMENTS.json").write_text(json.dumps(envs, indent=2), encoding="utf-8")
    bad = [k for k, v in src.items() if v.get("status") != "OK"]
    badp = [k for k, v in pins.items() if isinstance(v, dict) and v.get("matches") is False]
    print(json.dumps({"sources_bad": bad, "pins_bad": badp,
                      "envs": {k: v.get("exists") for k, v in envs.items() if k != "_system"}}, indent=2))
    print("IDENTITY_FREEZE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

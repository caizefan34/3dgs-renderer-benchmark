#!/usr/bin/env python3
"""CG0 identity gate + environment emitter (prereg section 1, fail-closed)."""
import os, sys, json, hashlib, platform, time
sys.path.insert(0, "/root/epic05_n1")
import cg0_prereg as P

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    import torch
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_mem_MB": round(torch.cuda.get_device_properties(0).total_memory/1e6) if torch.cuda.is_available() else None,
        "host": platform.node(),
        "env": "EPIC-05 A100 (authoritative)",
    }
    with open(os.path.join(out, "CG0_ENVIRONMENT.json"), "w") as f:
        json.dump(env, f, indent=2)

    # ---- identity gate ----
    id_pass = True
    detail = {}
    for cell in P.CELLS:
        scene = P.CELL_SCENE[cell]; it = P.CELL_ITER[cell]
        ck = P.ckpt_path(scene, it)
        disk = sha256_file(ck)
        prov = P.prov_path(scene)
        with open(prov) as f:
            pd = json.load(f)
        # collect all sha256 in provenance
        shas = set()
        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k.lower() in ("sha256", "sha256_hash", "hash") and isinstance(v, str) and len(v) == 64:
                        shas.add(v.lower())
                    walk(v)
            elif isinstance(o, list):
                for v in o: walk(v)
        walk(pd)
        match = disk.lower() in shas
        detail[cell] = {"disk": disk, "match": match, "N": None}
        # N
        ckpt = torch.load(ck, map_location="cpu", weights_only=False)
        detail[cell]["N"] = int(ckpt["xyz"].shape[0])
        detail[cell]["active_sh"] = int(ckpt["active_sh_degree"])
        if not match:
            id_pass = False
    # bicycle@25K live rule: hash must resolve (packet had null)
    b25 = detail["bicycle@25000"]
    if not b25["match"]:
        id_pass = False

    # binary pins
    bins = {
        "c0_ext": "/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so",
        "core_gsplat_cuda": "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so",
        "fused_ssim": "/root/miniforge3/envs/gsplat/lib/python3.10/site-packages/fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so",
    }
    bin_detail = {}
    for name, path in bins.items():
        if os.path.exists(path):
            h = sha256_file(path)
            bin_detail[name] = {"path": path, "sha256": h}
        else:
            bin_detail[name] = {"path": path, "sha256": None}
            id_pass = False

    result = {
        "cell": "ALL",
        "source_identity_pass": id_pass,
        "checkpoints": detail,
        "binary_pins": bin_detail,
        "authority": "N1 canonical trajectory + HM0 frozen environment",
    }
    with open(os.path.join(out, "CG0_SOURCE_IDENTITY.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"source_identity_pass": id_pass,
                      "bicycle25k_match": b25["match"],
                      "bins": {k: v["sha256"] for k, v in bin_detail.items()}}, indent=2))
    if not id_pass:
        print("CG0_BLOCKED_SOURCE_IDENTITY", flush=True)
        sys.exit(3)

if __name__ == "__main__":
    main()

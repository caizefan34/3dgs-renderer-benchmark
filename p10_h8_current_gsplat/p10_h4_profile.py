#!/usr/bin/env python3
"""P10 Gate H4 - mechanism profile for the H8-MR moment-space adjoint.

ncu hardware counters are NOT available on this host (driver resource blocked;
see --probe output). H4 therefore uses three counter-free evidence sources:

  1. static  cuobjdump -res-usage  -> registers / smem / spills per instantiation
  2. static  cuobjdump -sass       -> instruction census of the two kernels
                                      (off vs on template instantiation)
  3. dynamic torch.profiler        -> per-kernel CUDA time table, off vs on

modes
  single    --arm off|on --scene S            one fwd+bwd (for external tools)
  torchprof --scene S --out F                 per-kernel time table both arms
  sass      --so PATH --arm X --out F         resource usage + instruction census
  ncu_probe                                   record that counters are unavailable
  parse     --scene S --prof F --sass-off F --sass-on F --probe F --out JSON
"""
import argparse
import collections
import importlib.util
import json
import os
import re
import subprocess
import sys

REPO = "/root/3dgs-renderer-benchmark"
FROZEN = "/root/epic05_frozen"
ROOT = "/root/epic05_p10"
BINARIES = {
    "pristine": (f"{ROOT}/tree_pristine", f"{ROOT}/build/gsplat_cuda_p10_pristine/gsplat_cuda_p10_pristine.so"),
    "h8": (f"{ROOT}/tree_h8", f"{ROOT}/build/gsplat_cuda_p10_h8/gsplat_cuda_p10_h8.so"),
}
SEED = 4200
KERNELS = ["rasterize_to_pixels_3dgs_bwd_kernel", "projection_ewa_3dgs_fused_bwd_kernel",
           "rasterize_to_pixels_3dgs_fwd_kernel", "projection_ewa_3dgs_fused_fwd_kernel"]


def _load(binary):
    import torch  # noqa: F401  (loads libc10/libtorch into the process first)

    os.environ.pop("GSPLAT_H8_MR", None)
    tree, so = BINARIES[binary]
    modname = os.path.basename(so)[:-3]
    spec = importlib.util.spec_from_file_location(modname, so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[modname] = mod
    sys.modules["gsplat.csrc"] = mod
    sys.path.insert(0, tree)
    sys.path.insert(0, f"{REPO}/scripts/epic05/phase7")
    sys.path.insert(0, f"{REPO}/src")
    sys.path.insert(0, f"{REPO}/baseline/reference_v1")
    from gsplat import rasterization
    from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras
    return rasterization, load_ply, load_cameras_from_json, resize_cameras


def _inputs(load_ply, load_cams, resize_cams, scene):
    import torch
    ply = load_ply(f"{FROZEN}/p5_reference_plys/{scene}_30000.ply", device="cuda")
    cams = load_cams(f"{REPO}/data/official/mipnerf360/{scene}/cameras.json", device="cpu")
    c0 = cams[0]
    s = 2048.0 / max(c0.image_width, c0.image_height)
    cams = resize_cams(cams, int(round(c0.image_width * s / 2)) * 2,
                       int(round(c0.image_height * s / 2)) * 2)
    cam = cams[0]
    for a in ("viewmatrix", "K"):
        setattr(cam, a, getattr(cam, a).to("cuda"))
    means = ply["xyz"].clone().requires_grad_(True)
    quats = torch.nn.functional.normalize(ply["rotations"].clone(), dim=-1).requires_grad_(True)
    scales = torch.exp(ply["scales"].clone()).requires_grad_(True)
    opac = torch.sigmoid(ply["opacity"].clone()).requires_grad_(True)
    sh = ply["shs"].clone().contiguous().requires_grad_(True)
    g = torch.Generator(device="cuda").manual_seed(SEED)
    V = torch.randn((cam.image_height, cam.image_width, 3), generator=g, device="cuda")
    return dict(means=means, quats=quats, scales=scales, opacities=opac, colors=sh), cam, V, int(means.shape[0])


def _fwd(rasterization, p, cam):
    r, _, _ = rasterization(
        means=p["means"], quats=p["quats"], scales=p["scales"], opacities=p["opacities"],
        colors=p["colors"], viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height, tile_size=16, packed=False,
        sh_degree=3, radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=False)
    return r[0]


def cmd_single(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.arm == "on":
        os.environ["GSPLAT_H8_MR"] = "1"
    rasterization, load_ply, load_cams, resize_cams = _load(args.binary)
    import torch
    p, cam, V, n = _inputs(load_ply, load_cams, resize_cams, args.scene)
    for _ in range(args.warmup):
        (_fwd(rasterization, p, cam) * V).sum().backward()
        for t in p.values():
            t.grad = None
    torch.cuda.synchronize()
    (_fwd(rasterization, p, cam) * V).sum().backward()
    torch.cuda.synchronize()
    print(f"SINGLE {args.binary}/{args.arm}/{args.scene} N={n}")


def cmd_torchprof(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    rasterization, load_ply, load_cams, resize_cams = _load(args.binary)
    import torch
    p, cam, V, n = _inputs(load_ply, load_cams, resize_cams, args.scene)

    def set_arm(arm):
        if arm == "on":
            os.environ["GSPLAT_H8_MR"] = "1"
        else:
            os.environ.pop("GSPLAT_H8_MR", None)

    for arm in ("off", "on"):
        set_arm(arm)
        for _ in range(3):
            for t in p.values():
                t.grad = None
            (_fwd(rasterization, p, cam) * V).sum().backward()
    torch.cuda.synchronize()

    tables = {}
    for arm in ("off", "on"):
        set_arm(arm)
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False,
                     profile_memory=False) as prof:
            for _ in range(3):
                for t in p.values():
                    t.grad = None
                (_fwd(rasterization, p, cam) * V).sum().backward()
            torch.cuda.synchronize()
        agg = collections.defaultdict(lambda: {"cuda_time_us": 0.0, "count": 0})
        for ev in prof.key_averages():
            if ev.device_type is not None and str(ev.device_type) == "DeviceType.CUDA" and ev.self_device_time_total > 0:
                agg[ev.key]["cuda_time_us"] += float(ev.self_device_time_total)
                agg[ev.key]["count"] += int(ev.count)
        tables[arm] = {k: {"cuda_ms_total": v["cuda_time_us"] / 1000.0,
                           "cuda_ms_per_iter": v["cuda_time_us"] / 1000.0 / 3.0,
                           "count": v["count"]} for k, v in agg.items()}
    payload = {"binary": args.binary, "scene": args.scene, "n_gaussians": n,
               "iters": 3, "kernels": tables,
               "note": "CUPTI activity tracing via torch.profiler; per-kernel self CUDA time"}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"TORCHPROF {args.scene} -> {args.out}")


OP_RE = re.compile(r"^\s+/\*[0-9a-f]{4,}\*/\s+@?!?P?[0-9]?\s*([A-Z][A-Z0-9._]*)")
RES_FUNC_RE = re.compile(r"\s*Function (\S+):\s*$")
SASS_FUNC_RE = re.compile(r"\s*Function : (\S+)")
USAGE_RE = re.compile(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)")


def _select_names(res_lines):
    """Device function names belonging to the four kernels of interest."""
    names = []
    for ln in res_lines:
        m = RES_FUNC_RE.match(ln)
        if m and any(k in m.group(1) for k in KERNELS):
            names.append(m.group(1))
    return names


def _res_usage(res_lines):
    out = {}
    cur = None
    for ln in res_lines:
        m = RES_FUNC_RE.match(ln)
        if m:
            cur = m.group(1)
            continue
        m = USAGE_RE.search(ln)
        if m and cur:
            out[cur] = {"regs": int(m.group(1)), "stack": int(m.group(2)),
                        "shared": int(m.group(3)), "local": int(m.group(4))}
    return out


def _census(stream):
    """Opcode census, streamed: cuobjdump -sass on the full .so emits ~3.5 GB,
    so the caller must never buffer it (only the four kernel families are kept)."""
    funcs = {}
    cur = None
    for line in stream:
        m = SASS_FUNC_RE.match(line)
        if m:
            cur = m.group(1)
            funcs.setdefault(cur, collections.Counter())
            continue
        if cur is None:
            continue
        m = OP_RE.match(line)
        if m:
            funcs[cur][m.group(1).split(".")[0]] += 1
    return funcs


def cmd_sass(args):
    cuobjdump = "/usr/local/cuda/bin/cuobjdump"
    if not os.path.exists(cuobjdump):
        cuobjdump = "cuobjdump"
    res = subprocess.run([cuobjdump, "-arch", "sm_80", "-res-usage", args.so],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"cuobjdump -res-usage failed: {res.stderr[-2000:]}")
    res_lines = [l for l in res.stdout.splitlines() if l.strip()]
    names = _select_names(res_lines)
    if not names:
        raise RuntimeError("no matching kernel families found in .so")
    usage = _res_usage(res_lines)

    # restrict the SASS dump to the selected mangled names: the unfiltered dump
    # of this .so is ~3.5 GB (23 channel instantiations x several kernels)
    cmd = [cuobjdump, "-arch", "sm_80", "-sass"]
    for n in names:
        cmd += ["-fun", n]
    cmd.append(args.so)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    funcs = _census(p.stdout)
    p.stdout.close()
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"cuobjdump -sass failed with rc={rc}")

    def find_kernel(name, arm):
        """Instantiation selection: patched .so carries Lb0E/Lb1E (bool H8_MR);
        the pristine .so has no bool template argument at all."""
        hits = []
        for f, ops in funcs.items():
            if name not in f:
                continue
            if arm == "on" and "Lb1E" in f:
                hits.append((f, ops))
            elif arm == "off" and "Lb0E" in f:
                hits.append((f, ops))
            elif arm == "off" and "Lb0E" not in f and "Lb1E" not in f:
                hits.append((f, ops))  # pristine instantiation (no bool param)
        return hits

    out = {"so": args.so, "arm": args.arm, "n_device_functions_matched": len(names),
           "sass_census_note": "SASS dump filtered with cuobjdump -fun to the four "
                               "kernel families (full dump is ~3.5 GB)",
           "resource_usage_by_func": {n: usage.get(n) for n in names},
           "kernels": {}}
    for name in KERNELS:
        for f, ops in find_kernel(name, args.arm):
            total = sum(ops.values())
            out["kernels"][f] = {
                "total_instructions": total,
                "fp_ops": {k: ops.get(k, 0) for k in
                           ("FFMA", "FADD", "FMUL", "FADDF", "FSETP", "FSEL", "FMNMX", "MUFU", "FFMA32I")},
                "fp_total": sum(ops.get(k, 0) for k in ("FFMA", "FADD", "FMUL", "FADDF", "FSETP", "FSEL", "FMNMX")),
                "atom_red": {k: ops.get(k, 0) for k in ("RED", "ATOM", "ATOMG", "REDG", "ATOMS", "REDS")},
                "load_store": {k: ops.get(k, 0) for k in ("LDG", "STG", "LDS", "STS", "LDL", "STL")},
                "regs": (usage.get(f) or {}).get("regs"),
                "spill_local": (usage.get(f) or {}).get("local"),
                "spill_stack": (usage.get(f) or {}).get("stack"),
                "top_opcodes": dict(ops.most_common(25)),
            }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"SASS {args.arm} instantiations={len(out['kernels'])} -> {args.out}")


def cmd_ncu_probe(args):
    out = {}
    for cmd in (["/usr/local/cuda/bin/ncu", "--query-metrics"],
                ["/usr/local/cuda/bin/ncu", "--metrics", "gpu__time_duration.sum,launch__registers_per_thread",
                 "-k", "regex:elementwise", "--launch-count", "1", sys.executable, "-c",
                 "import torch;a=torch.randn(1000,1000,device='cuda');(a+a).sum().item()"]):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)})
        out[" ".join(cmd[:2])] = {"returncode": r.returncode,
                                  "stdout_tail": r.stdout[-1500:], "stderr_tail": r.stderr[-1500:]}

    def blocked(v):
        blob = v["stdout_tail"] + v["stderr_tail"]
        return (v["returncode"] != 0
                or "==ERROR==" in blob
                or "driver resource was unavailable" in blob
                or "No devices found" in blob)
    reasons = [k for k, v in out.items() if blocked(v)]
    out["conclusion"] = ("hardware counters unavailable on this host" if reasons else "counters available")
    out["blocked_commands"] = reasons
    out["driver_resource_note"] = (
        "ncu reports 'a driver resource was unavailable' and advises that no other tool "
        "(e.g. DCGM) may collect profiling data concurrently; the GPU nodes on this host "
        "run a datacenter GPU manager, so the counter measurement library fails to attach")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"NCU_PROBE: {out['conclusion']} -> {args.out}")


def cmd_parse(args):
    with open(args.prof) as f:
        prof = json.load(f)
    out = {"scene": prof["scene"], "n_gaussians": prof["n_gaussians"],
           "hardware_counters": None, "method": {}}
    with open(args.probe) as f:
        out["hardware_counters_ncu_probe"] = json.load(f)
        out["hardware_counters"] = "unavailable (see probe)"
    out["method"]["dynamic_kernel_times"] = "torch.profiler CUPTI activity tracing, 3 iters/arm, warmup 3"
    out["method"]["static"] = "cuobjdump -res-usage / -sass on the two .so"

    for arm, path in (("off", args.sass_off), ("on", args.sass_on)):
        with open(path) as f:
            out[f"sass_{arm}"] = json.load(f)

    # per-kernel time delta table
    koff = prof["kernels"]["off"]
    kon = prof["kernels"]["on"]
    rows = []
    for k in sorted(set(koff) | set(kon), key=lambda x: -(koff.get(x, {}).get("cuda_ms_per_iter", 0))):
        a = koff.get(k, {}).get("cuda_ms_per_iter")
        b = kon.get(k, {}).get("cuda_ms_per_iter")
        rows.append({"kernel": k, "off_ms": a, "on_ms": b,
                     "delta_ms": (b - a) if (a is not None and b is not None) else None,
                     "delta_pct": ((b - a) / a * 100) if (a and b is not None) else None})
    out["kernel_time_table"] = rows
    out["total_cuda_ms_per_iter"] = {
        "off": sum(v["cuda_ms_per_iter"] for v in koff.values()),
        "on": sum(v["cuda_ms_per_iter"] for v in kon.values()),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"PROFILE -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("single")
    s.add_argument("--binary", choices=sorted(BINARIES), default="h8")
    s.add_argument("--arm", choices=["off", "on"], required=True)
    s.add_argument("--scene", default="room")
    s.add_argument("--gpu", type=int, default=0)
    s.add_argument("--warmup", type=int, default=2)
    s.set_defaults(func=cmd_single)
    t = sub.add_parser("torchprof")
    t.add_argument("--binary", choices=sorted(BINARIES), default="h8")
    t.add_argument("--scene", required=True)
    t.add_argument("--gpu", type=int, default=0)
    t.add_argument("--out", required=True)
    t.set_defaults(func=cmd_torchprof)
    a = sub.add_parser("sass")
    a.add_argument("--so", required=True)
    a.add_argument("--arm", choices=["off", "on"], required=True)
    a.add_argument("--out", required=True)
    a.set_defaults(func=cmd_sass)
    p = sub.add_parser("ncu_probe")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_ncu_probe)
    q = sub.add_parser("parse")
    q.add_argument("--prof", required=True)
    q.add_argument("--sass-off", required=True)
    q.add_argument("--sass-on", required=True)
    q.add_argument("--probe", required=True)
    q.add_argument("--out", required=True)
    q.set_defaults(func=cmd_parse)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

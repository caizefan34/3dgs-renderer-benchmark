#!/usr/bin/env python3
"""Build an H4 static block (the schema p10_verdict.py reads) from a full-stream
SASS extraction directory.

Unlike `p10_h4_profile.py sass`, which drives `cuobjdump -fun`, this works from
p10_sass_extract.py's output and therefore covers the projection-backward
instantiations, whose mangled names `-fun` cannot resolve.
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, "/root/epic05_p10")
from p10_sass_cmp import FP_OPS, INSTR_RE, FUNC_RE, scalar_of  # noqa: E402

RES_USAGE_RE = re.compile(r"^\s*Function (\S+):\s*$")
USAGE_RE = re.compile(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)")


def res_usage(so):
    import subprocess
    out = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "-arch", "sm_80", "-res-usage", so],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    usage, cur = {}, None
    for ln in out.stdout.splitlines():
        m = RES_USAGE_RE.match(ln)
        if m:
            cur = m.group(1)
            continue
        m = USAGE_RE.search(ln)
        if m and cur:
            usage[cur] = {"regs": int(m.group(1)), "stack": int(m.group(2)),
                          "shared": int(m.group(3)), "local": int(m.group(4))}
    return usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--so", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--marker", default=None,
                    help="bool-template marker to keep (Lb0E/Lb1E). Kernels that carry no "
                         "marker at all (the un-patched forward kernels) are kept in every "
                         "arm, since they have a single instantiation shared by both arms.")
    ap.add_argument("--families", default="rasterize_to_pixels_3dgs_bwd_kernel,"
                                         "projection_ewa_3dgs_fused_bwd_kernel,"
                                         "rasterize_to_pixels_3dgs_fwd_kernel,"
                                         "projection_ewa_3dgs_fused_fwd_kernel")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fams = tuple(a.families.split(","))

    usage = res_usage(a.so)
    kernels = {}
    for fn in sorted(os.listdir(a.dir)):
        if not fn.endswith(".sass"):
            continue
        name = None
        ops = collections.Counter()
        with open(os.path.join(a.dir, fn)) as f:
            for line in f:
                m = FUNC_RE.match(line)
                if m:
                    name = m.group(1)
                    continue
                m = INSTR_RE.match(line)
                if m:
                    ops[m.group(2).split(".")[0]] += 1
        if name is None or not any(f in name for f in fams):
            continue
        if a.marker:
            has_any = ("Lb0E" in name) or ("Lb1E" in name)
            if has_any and a.marker not in name:
                continue
        usage_rec = next((v for k, v in usage.items() if k == name), None)
        fp_ops = {k: ops.get(k, 0) for k in ("FFMA", "FADD", "FMUL", "FADDF", "FSETP",
                                             "FSEL", "FMNMX", "MUFU", "FFMA32I")}
        kernels[name] = {
            "total_instructions": sum(ops.values()),
            "fp_ops": fp_ops,
            "fp_total": sum(ops.get(k, 0) for k in ("FFMA", "FADD", "FMUL", "FADDF",
                                                    "FSETP", "FSEL", "FMNMX")),
            "atom_red": {k: ops.get(k, 0) for k in ("RED", "ATOM", "ATOMG", "REDG", "ATOMS", "REDS")},
            "load_store": {k: ops.get(k, 0) for k in ("LDG", "STG", "LDS", "STS", "LDL", "STL")},
            "regs": (usage_rec or {}).get("regs"),
            "spill_local": (usage_rec or {}).get("local"),
            "spill_stack": (usage_rec or {}).get("stack"),
            "top_opcodes": dict(ops.most_common(25)),
        }
    out = {"so": a.so, "arm": a.arm, "source": f"full-stream extraction of {a.dir}",
           "marker": a.marker, "n_kernels": len(kernels), "kernels": kernels}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    fam_counts = collections.Counter()
    for k in kernels:
        for fam in fams:
            if fam in k:
                fam_counts[fam] += 1
    print(f"H4 static arm={a.arm} marker={a.marker} kernels={len(kernels)} "
          f"{dict(fam_counts)} -> {a.out}")


if __name__ == "__main__":
    main()

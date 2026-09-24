#!/usr/bin/env python3
"""TASK 1: resource + SASS audit for higs_blend_bwd_px_kernel in two .so (sm_80)."""
import json, re, subprocess, sys
from collections import Counter

CUBIN = None
def cuobjdump(args):
    return subprocess.run(["cuobjdump"] + list(args), capture_output=True, text=True).stdout

def find_blend_kernel(sass):
    # capture function-delimited sections and pick the blend_bwd_px one
    sections = re.split(r"\n(?=\s*Function : )", sass)
    for sec in sections:
        if "higs_blend_bwd_px_kernel" in sec.splitlines()[0] if sec.splitlines() else "":
            return sec
    return None

def resource(sass):
    out = {}
    for line in sass.splitlines():
        m = re.match(r"\s*(REG|STACK|SHARED|LOCAL|MAXNTID):\s*(\d[\d, ]*)", line)
        if m:
            out[m.group(1)] = int(m.group(2).replace(",", "").strip())
    return out

def histogram(kernsec):
    ops = Counter()
    total = 0
    for line in kernsec.splitlines():
        m = re.match(r'\s*/\*[0-9a-f]+\*/\s*(.*?);?\s*$', line)
        if not m:
            continue
        toks = m.group(1).split()
        if not toks:
            continue
        # drop branch predicate tokens like @!P0, @P1
        i = 0
        while i < len(toks) and toks[i].startswith("@"):
            i += 1
        if i >= len(toks):
            continue
        op = toks[i]
        # pseudo / data lines: first token is a register or memory -> skip as non-op
        if re.match(r"^(R\d|U\d|RZ|URZ|UR\d|c\[|shared|lmem|sld|#)", op):
            continue
        ops[op] += 1
        total += 1
    return total, ops

def norm_hist(ops, total):
    return {k: round(v / total, 5) for k, v in sorted(ops.items())} if total else {}

for label, so in [("PROD","/mnt/storage_pool/liaoyuanjun/higs_c0_cache_composed/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"),
                  ("V0","/mnt/storage_pool/liaoyuanjun/higs_p3h_cache/V0/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so")]:
    ru = cuobjdump(["--dump-resource-usage", so])
    sass = cuobjdump(["--dump-sass", so])
    # resources for blend kernel across all its resource sections
    res = {}
    for sec in re.split(r"\n(?=.*Function:)", ru) if False else re.split(r"\n *(?=arch)", ru):
        pass
    # simpler: gather all resource blocks, keep those whose preceding function name contains blend
    blocks = re.split(r"\n(?=\s*Function:)|\n(?=Function:)", ru)
    for blk in blocks:
        if "higs_blend_bwd_px_kernel" in blk:
            r = resource(blk)
            for k, v in r.items():
                res.setdefault(k, v)
    ksec = find_blend_kernel(sass)
    total = None; ops = None
    if ksec is not None:
        total, ops = histogram(ksec)
    print("===== %s =====" % label)
    print("RESOURCE(blend_bwd_px) ", res)
    print("SASS instructions(kernel) ", total)
    if ops is not None:
        print("SASS opcode histogram:")
        for op, c in sorted(ops.items(), key=lambda x: (-x[1], x[0])):
            print("  %-22s %6d" % (op, c))
    print()

print("AUDIT_DONE")
#!/usr/bin/env python3
"""Restore the trailing backslashes on the two sink-argument lines inside the
HIGS_LAUNCH_BLEND_BWD_PX macro. Multi-line macro bodies require a trailing '\\'
on every line except the last. Patch #3 previously dropped them."""
F = "/mnt/storage_pool/liaoyuanjun/higs_p3h_worktree_gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu"
lines = open(F, encoding="utf-8").read().split("\n")

def fix(a, b):
    global lines
    # locate both lines in order
    ia = None
    for i, l in enumerate(lines):
        if l.lstrip() == a:
            ia = i
            break
    assert ia is not None, "missing line: %r" % a
    ib = None
    for i in range(ia + 1, len(lines)):
        if lines[i].lstrip() == b:
            ib = i
            break
    assert ib is not None, "missing line after %r: %r" % (a, b)
    ind = lines[ia][:len(lines[ia]) - len(lines[ia].lstrip())]
    lines[ia] = ind + a + "\\"
    lines[ib] = ind + b + "\\"
    print("  fixed line %d -> %s" % (ia + 1, lines[ia].rstrip()))
    print("  fixed line %d -> %s" % (ib + 1, lines[ib].rstrip()))

fix("v_backgrounds.data_ptr<float>(),", "v_p3h_scratch.data_ptr<float>()")

open(F, "w", encoding="utf-8").write("\n".join(lines))
print("BACKSLASH RESTORED OK")

# verify: all sane macro lines contiguous
src = open(F, encoding="utf-8").read().split("\n")
for i in range(1500, 1520):
    print("%5d: %s" % (i + 1, src[i].rstrip()))
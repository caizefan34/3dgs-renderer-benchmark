#!/usr/bin/env python3
"""Stream the full `cuobjdump -arch sm_80 -sass` dump of a .so (multi-GB) and
keep only the backward kernel families of interest.

`cuobjdump -fun <mangled>` cannot resolve the projection-backward symbols, so the
projection bwd SASS was missing from the earlier census-based comparison. This
walks the whole dump once and writes one text file per matched function.
"""
import hashlib
import os
import re
import subprocess
import sys

FAMS = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else (
    "rasterize_to_pixels_3dgs_bwd_kernel", "projection_ewa_3dgs_fused_bwd_kernel",
    "projection_ewa_3dgs_fused_fwd_kernel", "rasterize_to_pixels_3dgs_fwd_kernel")
SASS_FUNC_RE = re.compile(r"\s*Function : (\S+)")


def main():
    so, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    p = subprocess.Popen(["/usr/local/cuda/bin/cuobjdump", "-arch", "sm_80", "-sass", so],
                         stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    cur, fh, counts = None, None, {}
    for line in p.stdout:
        m = SASS_FUNC_RE.match(line)
        if m:
            name = m.group(1)
            if fh:
                fh.close()
                fh = None
            cur = name if any(f in name for f in FAMS) else None
            if cur:
                path = os.path.join(outdir, hashlib.sha1(cur.encode()).hexdigest()[:12] + ".sass")
                fh = open(path, "w")
                fh.write(f"Function : {cur}\n")
                counts[cur] = counts.get(cur, 0) + 1
            continue
        if cur and fh:
            fh.write(line)
    if fh:
        fh.close()
    p.stdout.close()
    rc = p.wait()
    print(f"rc={rc} matched_functions={len(counts)} outdir={outdir}")
    for k in sorted(counts):
        print(counts[k], k)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Targeted resource + SASS diff for the exact blend_bwd_px_kernel<3,2,true,true>
instance in both .so. Accepts a template-mangling regex core for CDIM=3,PX=2,t=true,t=true.
cuobjdump resource-usage uses ILj3E..., --dump-sass uses ILi3E..."""
import re, subprocess
from collections import Counter

FRAG = r"higs_blend_bwd_px_kernelIL[ji]3EL[ji]2ELb1ELb1E"
SO = {
 "PROD":"/mnt/storage_pool/liaoyuanjun/higs_c0_cache_composed/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so",
 "V0":"/mnt/storage_pool/liaoyuanjun/higs_p3h_cache/V0/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so",
}
rx = re.compile(FRAG)

def cmd(args):
    return subprocess.run(["cuobjdump"]+list(args), capture_output=True, text=True).stdout

def resource_for(ru):
    cur=None; out=None
    for line in ru.splitlines():
        if "Function " in line and rx.search(line):
            out={}
        elif out is not None and "REG:" in line:
            m=re.match(r"\s*REG:(\S+) STACK:(\S+) SHARED:(\S+) LOCAL:(\S+)", line)
            if m:
                out={"REG":m.group(1),"STACK":m.group(2),"SHARED":m.group(3),"LOCAL":m.group(4)}
                return out
    return out

def sass_sec(sass):
    # split into Function sections
    parts=re.split(r"\n(?=\s*Function : )", sass)
    for p in parts:
        head=p.split("\n",1)[0]
        if rx.search(head):
            return p
    return None

def hist(sec):
    ops=Counter(); tot=0
    for line in sec.splitlines():
        m=re.match(r'\s*/\*[0-9a-f]{4}\*/\s*(.*)', line)
        if not m: continue
        t=m.group(1).split()
        if not t: continue
        i=0
        while i<len(t) and t[i].startswith("@"): i+=1
        if i>=len(t): continue
        op=t[i]
        if re.match(r"^(R\d|U\d|RZ|URZ|UR\d|c\[|shared|lmem|sld|#|s\[)", op): continue
        ops[op]+=1; tot+=1
    return tot, ops

def dump(label):
    ru=cmd(["--dump-resource-usage", SO[label]])
    sass=cmd(["--dump-sass", SO[label]])
    r=resource_for(ru)
    sec=sass_sec(sass)
    print("===== %s  blend_bwd_px_kernel<3,2,true,true> =====" % label)
    print("RESOURCE:", r if r else "NOT FOUND (pool may only contain it via inlined/pooled variant)")
    if sec is None:
        print("SASS: function not present as standalone"); return r,None
    tot,ops=hist(sec)
    print("SASS instrs:", tot)
    return r,(tot,ops)

rp,(tp,op)=dump("PROD")
rv,(tv,ov)=dump("V0")
print("\n=== normalized opcode histogram diff (nondominant) ===")
allkeys=set(op)|set(ov)
diffs=[]
for k in allkeys:
    cp=op.get(k,0)/tp if tp else 0.0
    cv=ov.get(k,0)/tv if tv else 0.0
    if abs(cp-cv)>1e-9:
        diffs.append((k,op.get(k,0),ov.get(k,0)))
if not diffs:
    print("IDENTICAL histograms across %d opcodes; total %s vs %s (PROD/V0)"%(len(allkeys),tp,tv))
else:
    for k,cp_,cv_ in diffs:
        print("  %-24s PROD %5d  V0 %5d"%(k,cp_,cv_))
print("DONE_TASK1_TARGETED")
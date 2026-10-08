import hashlib, json, os
OUT="/root/gsc0_out"; RUN="/root/gsc0_run"
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()
ti=json.load(open(os.path.join(OUT,"GSC0_TRACE_INDEX.json")))
n_ok=0; n_bad=0
for scene,entry in ti["scenes"].items():
    for rec in entry.get("trace_records",[]):
        p=os.path.join(RUN,scene,"traces",rec["file"])
        h=sha(p)
        if h==rec["sha256"]: n_ok+=1
        else: n_bad+=1; print("BAD",p)
# also verify manifest raw entries
man=open(os.path.join(OUT,"MANIFEST.sha256")).read().splitlines()
man_raw={}
for l in man:
    if "  raw/" in l:
        h,name=l.split("  ",1); man_raw[name]=h
print("trace_index raw verified: ok=%d bad=%d"%(n_ok,n_bad))
print("manifest raw entries:",len(man_raw))
# manifest raw path raw/<scene>/traces/<file> -> RUN/<scene>/traces/<file>
m_ok=0
for name,h in man_raw.items():
    rel=name[len("raw/"):]
    p=os.path.join(RUN,rel)
    if sha(p)==h: m_ok+=1
    else: print("MANIFEST BAD",name)
print("manifest raw verified: ok=%d/%d"%(m_ok,len(man_raw)))
print(">>> FULL VERIFICATION", "PASS" if (n_bad==0 and n_ok==9 and m_ok==len(man_raw)) else "FAIL")
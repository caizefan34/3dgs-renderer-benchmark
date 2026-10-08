import json, glob, os
for f in sorted(glob.glob("/root/epic05_p17/*.json")) + sorted(glob.glob("/root/epic05_p13x/**/*.json", recursive=True)):
    try:
        d=json.load(open(f))
    except: continue
    s=json.dumps(d)
    if "total_ms_mean" in s or "full_run" in s:
        print("TIMING", f)
        fr=d.get("full_run",{})
        print("  full_run:", {k:round(v,2) for k,v in fr.items() if isinstance(v,(int,float))})
        break
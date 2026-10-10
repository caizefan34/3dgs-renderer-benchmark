import glob, os, time
now=time.time()
for f in sorted(glob.glob("/dev/shm/wv0_out/logs/*.log")):
    sid=os.path.basename(f).replace(".log","")
    with open(f,errors="replace") as fh:
        lines=[l for l in fh.read().splitlines() if l.strip()]
    last=lines[-1] if lines else "(empty)"
    # last G= progress
    print(f"{sid:24s} | {last[:90]}")
print("=== produced ===")
for f in sorted(glob.glob("/dev/shm/wv0_out/states/*.pt")):
    print(os.path.basename(f), round(os.path.getsize(f)/1e6,2),"MB", "age", round((now-os.path.getmtime(f))/60,1),"min")

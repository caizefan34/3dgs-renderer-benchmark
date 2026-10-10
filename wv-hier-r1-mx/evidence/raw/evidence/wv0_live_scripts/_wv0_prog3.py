import glob, os, time
now=time.time()
for f in sorted(glob.glob("/dev/shm/wv0_out/states/*.json.log")):
    sid=os.path.basename(f).replace(".json.log","")
    age=now-os.path.getmtime(f)
    lines=[l for l in open(f,errors="replace").read().splitlines() if l.strip()]
    gline = next((l for l in reversed(lines) if "] G=" in l), None)
    nlines=len(lines)
    print(f"{sid:24s} age={age:5.0f}s lines={nlines:4d} | {gline[-55:] if gline else '(no G line yet)'}")
print("done:", len(glob.glob("/dev/shm/wv0_out/states/*.pt")))

import glob, os, time
now=time.time()
for f in sorted(glob.glob("/dev/shm/wv0_out/states/*.json.log")):
    sid=os.path.basename(f).replace(".json.log","")
    lines=[l for l in open(f,errors="replace").read().splitlines() if l.strip()]
    # last G= progress line
    gline = next((l for l in reversed(lines) if l.split("]")[-1].strip().startswith("G=")), None)
    print(f"{sid:24s} | {gline[-60:] if gline else (lines[-1][-60:] if lines else '(empty)')}")

import hashlib, glob, os
roots = ["/root/hm0_pkg", "/root/epic05_moonshot/a0"]
seen = {}
for root in roots:
    for f in glob.glob(root + "/**/*.so", recursive=True):
        try:
            h = hashlib.sha256(open(f, "rb").read()).hexdigest()
        except Exception as e:
            print("ERR", f, e); continue
        seen[f] = h
for f in sorted(seen):
    print(seen[f][:16], f)
print("TOTAL_SO", len(seen))

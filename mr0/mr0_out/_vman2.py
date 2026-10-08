import hashlib
ok = True
for line in open("MANIFEST.sha256"):
    h, name = line.split()
    local = hashlib.sha256(open(name, "rb").read()).hexdigest()
    if local != h:
        ok = False; print("MISMATCH", name)
print("ALL_OK" if ok else "FAILED")

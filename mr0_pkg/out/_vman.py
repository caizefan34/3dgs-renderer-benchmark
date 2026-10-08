import hashlib, os
ok = True
for line in open("MANIFEST.sha256"):
    h, name = line.split()
    local = hashlib.sha256(open(name, "rb").read()).hexdigest()
    match = "OK" if local == h else "MISMATCH"
    if local != h: ok = False
    print(match, name)
print("ALL_OK" if ok else "FAILED")

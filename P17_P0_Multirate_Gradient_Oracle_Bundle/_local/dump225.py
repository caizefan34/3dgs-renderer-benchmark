data = open("/root/epic05_p17/p17_p0_oracle.py", "rb").read()
lines = data.split(b"\n")
ln = lines[224]  # 0-based -> line 225
print("len:", len(ln))
for j, b in enumerate(ln):
    ch = chr(b) if 32 <= b < 127 else "?"
    print(j, b, ch)

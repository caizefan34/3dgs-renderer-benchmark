p = r"C:\Users\36570\3dgs-renderer-benchmark\P17_P0_Multirate_Gradient_Oracle_Bundle\_local\p17_p0_oracle.py"
data = open(p, "rb").read()
# show every line that looks like a dict-return line with byte detail
for i, l in enumerate(data.split(b"\n")):
    if b"self.model._xyz" in l and l.startswith(b"        return"):
        print(i + 1, "len", len(l), repr(l[-12:]))
# count occurrences of the exact prefix
prefix = b'        return {"xyz": self.model._xyz'
print("count:", data.count(prefix))
idx = data.find(prefix)
print("first idx:", idx)
seg = data[idx:idx+90]
print("seg:", repr(seg))

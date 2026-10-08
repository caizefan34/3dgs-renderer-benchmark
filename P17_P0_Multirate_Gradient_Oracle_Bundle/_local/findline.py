import sys
lines = open(sys.argv[1], encoding="utf-8").read().split("\n")
target = '        return {"xyz": self.model._xyz, "shs": self.model._shs",'
hits = [i + 1 for i, l in enumerate(lines) if l == target]
print("exact matches:", hits)
near = [i + 1 for i, l in enumerate(lines) if "return {" in l and "_xyz" in l]
print("near matches:", near)
for n in near:
    print(n, repr(lines[n - 1]))

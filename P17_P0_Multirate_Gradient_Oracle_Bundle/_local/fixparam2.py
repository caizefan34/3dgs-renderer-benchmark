import re
p = r"C:\Users\36570\3dgs-renderer-benchmark\P17_P0_Multirate_Gradient_Oracle_Bundle\_local\p17_p0_oracle.py"
lines = open(p, "r", encoding="utf-8").read().split("\n")
out = []
i = 0
n_repl = 0
while i < len(lines):
    l = lines[i]
    if l == '        return {"xyz": self.model._xyz, "shs": self.model._shs,"':
        # replace this + next 3 lines with a single-line expression
        del lines[i:i + 4]
        lines.insert(i, '        return getattr(self.model, {"xyz": "_xyz", "shs": "_shs", "opacity": "_opacity", "scaling": "_scaling", "rotation": "_rotation"}[name])')
        n_repl += 1
    i += 1
with open(p, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(lines))
print("replaced:", n_repl)

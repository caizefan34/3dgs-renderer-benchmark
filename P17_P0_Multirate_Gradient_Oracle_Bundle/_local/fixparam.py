p = r"C:\Users\36570\3dgs-renderer-benchmark\P17_P0_Multirate_Gradient_Oracle_Bundle\_local\p17_p0_oracle.py"
lines = open(p, "r", encoding="utf-8").read().split("\n")
hits = [i for i, l in enumerate(lines) if l.startswith('        return {"xyz"')]
print("matches at (0-based):", hits)
for i in hits:
    new = (
        '        return getattr(self.model, {"xyz": "_xyz", "shs": "_shs",\n'
        '        "opacity": "_opacity", "scaling": "_scaling",\n'
        '        "rotation": "_rotation"}[name])'
    )
    lines[i] = new
with open(p, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(lines))
print("rewritten", len(hits), "occurrence(s)")

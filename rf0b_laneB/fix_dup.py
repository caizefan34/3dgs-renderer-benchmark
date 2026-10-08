p = "rf0b_fastgs_model.py"
lines = open(p, encoding="utf-8").read().split("\n")
# remove the duplicate z line at index 73 (line 74)
assert lines[72].strip().startswith("z = P[:, 2]"), lines[72]
assert lines[73].strip().startswith("z = P[:, 2]"), lines[73]
del lines[73]
open(p, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
print("removed duplicate z")
p = "rf0b_fastgs_model.py"
lines = open(p, encoding="utf-8").read().split("\n")
# Replace lines 72-73 (1-based) -> indices 71,72
assert "Xh2 = torch.cat" in lines[71], lines[71]
assert "P = (K @" in lines[72], lines[72]
lines[71] = "    P = (K @ Xv.T).T                                   # [N,3]  (K is 3x3: drop homog)"
lines[72] = "    z = P[:, 2].clamp_min(1e-8)"
# The old line 74 was "z = P[:, 2].clamp_min(1e-8)" already? check
print("L74:", repr(lines[73]))
print("L75:", repr(lines[74]))
print("L76:", repr(lines[75]))
open(p, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
print("done")
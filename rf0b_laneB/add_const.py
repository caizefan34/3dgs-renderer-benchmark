p = "rf0b_fastgs_model.py"
s = open(p, encoding="utf-8").read()
old = "FGS_OPACITY_CAP = 0.8         # post-densify opacity cap (vs 0.01 in reset_opacity)"
new = old + "\nFGS_N_SCORE_VIEWS = 10     # n_score_views (train.py compute_gaussian_score_fastgs camlist size)"
assert old in s and "FGS_N_SCORE_VIEWS" not in s
s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("added FGS_N_SCORE_VIEWS")
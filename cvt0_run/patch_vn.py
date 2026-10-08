p = "cvt0_run.py"
s = open(p, encoding="utf-8").read()
old = "sel_pos, sel_cams, panels, hpos, hold_panel = build_panels(train, hold)"
new = "sel_pos, sel_cams, panels, hpos, hold_panel = build_panels(train, holdout)"
assert old in s
s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched variable name hold->holdout")

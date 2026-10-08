p = "cvt0_run.py"
s = open(p, encoding="utf-8").read()
old = '''            rec = {
                "panel": pname, "panel_cams": panel_cams, "holdout_cams": hold_panel,
                "L0_panel": L0_panel, "L0_holdout": L0_hold,'''
new = '''            rec = {
                "panel": pname, "panel_cams": panel_cams, "holdout_cams": hold_panel,
                "L0_panel": L0_panel, "L0_holdout": L0_hold,
                "base_panel_eval": base_panel, "base_holdout_eval": base_hold,
                "post_panel_eval": post_panel, "post_holdout_eval": post_hold,'''
assert old in s
s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched rec to store full eval dicts")

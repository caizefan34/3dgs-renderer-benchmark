p = "cvt0_report.py"
s = open(p, encoding="utf-8").read()
old = "core `gsplat_cuda.so` = `{env['core_so_sha256'][:24]}...` \u2014 both match the"
new = "core `gsplat_cuda.so` = `{src['core_gsplat_so_sha256'][:24]}...` \u2014 both match the"
assert old in s, "env core line not found"
s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched report to use src core hash")

import io
p = "cvt0_run.py"
s = open(p, encoding="utf-8").read()
old = "PREREG_PIN_C0 = \"d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11\""
new = "PREREG_PIN_C0 = \"d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11\""
assert old in s, "old pin not found"
s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched PREREG_PIN_C0 to 64-hex authoritative value")

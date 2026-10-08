p = "cvt0_report.py"
s = open(p, encoding="utf-8").read()

old_hdr = '''| gate | strong thr | promote thr | r10 | r15 | r20 | b10 | b15 | b20 | (strong) |
|------|------|------|------|------|------|------|------|------|'''
new_hdr = '''| gate (strong / promote) | r10 | r15 | r20 | b10 | b15 | b20 |
|------|---|---|---|---|---|---|'''
assert old_hdr in s
s = s.replace(old_hdr, new_hdr)

old_row = '''    gate_rows.append(f"| {name} | {sthr} | {pthr} | " +
                     " | ".join(scells) + " | " + " | ".join(pcells) + " |")'''
new_row = '''    gate_rows.append(f"| {name} ({sthr}/{pthr}) | " +
                     " | ".join(scells) + " |")'''
assert old_row in s
s = s.replace(old_row, new_row)

open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched gate table columns")

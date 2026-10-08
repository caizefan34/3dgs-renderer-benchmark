import ast, sys
lines = open(sys.argv[1], encoding="utf-8").read().split("\n")
# binary search the largest prefix that compiles
lo, hi = 1, len(lines)
ok_at = 0
while lo <= hi:
    mid = (lo + hi) // 2
    try:
        ast.parse("\n".join(lines[:mid]))
        ok_at = mid
        lo = mid + 1
    except SyntaxError:
        hi = mid - 1
print("largest ast-parseable prefix:", ok_at)
for i in range(max(0, ok_at - 2), min(len(lines), ok_at + 3)):
    print(i + 1, repr(lines[i]))

import sys
lines = open(sys.argv[1], encoding="utf-8").read().split("\n")
for n in (150, 200, 220, 224, 225, 228, 300):
    try:
        src = "\n".join(lines[:n])
        compile(src, "x", "exec")
        print("n =", n, "OK")
    except SyntaxError as e:
        print("n =", n, "FAIL line", e.lineno, "col", e.offset, ":", e.msg)

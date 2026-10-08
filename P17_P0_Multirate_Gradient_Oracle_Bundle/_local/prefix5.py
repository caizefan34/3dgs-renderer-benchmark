import sys
lines = open(sys.argv[1], encoding="utf-8").read().split("\n")
for n in (350, 500, 700, len(lines)):
    src = "\n".join(lines[:n])
    try:
        compile(src, "x", "exec")
        print("n =", n, "OK")
    except SyntaxError as e:
        print("n =", n, "FAIL line", e.lineno, "col", e.offset, ":", e.msg)

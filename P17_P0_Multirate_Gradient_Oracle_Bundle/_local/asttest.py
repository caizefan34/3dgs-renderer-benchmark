import ast, sys
lines = open(sys.argv[1], encoding="utf-8").read().split("\n")
n = int(sys.argv[2])
try:
    ast.parse("\n".join(lines[:n]))
    print("n =", n, "OK")
except SyntaxError as e:
    print("n =", n, "FAIL line", e.lineno, "col", e.offset, ":", e.msg)

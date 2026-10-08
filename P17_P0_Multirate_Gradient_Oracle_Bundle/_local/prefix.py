import sys
lines = open(sys.argv[1], encoding="utf-8").read().split("\n")
n = int(sys.argv[2])
try:
    compile("\n".join(lines[:n]), "x", "exec")
    print("n =", n, "OK")
except SyntaxError as e:
    print("n =", n, "FAIL at line", e.lineno, ":", e.msg)

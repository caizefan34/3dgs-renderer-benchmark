import sys
lines = open(sys.argv[1], encoding="utf-8").read().split("\n")
# find the smallest n in [225..260] that compiles, if any
for n in range(225, 261):
    src = "\n".join(lines[:n])
    try:
        compile(src, "x", "exec")
        print("n =", n, "OK")
        break
    except SyntaxError:
        pass
else:
    print("no prefix 225..260 compiles")

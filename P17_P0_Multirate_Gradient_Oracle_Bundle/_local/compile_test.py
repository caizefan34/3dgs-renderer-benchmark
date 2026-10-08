import sys
src = open(sys.argv[1], encoding="utf-8").read()
try:
    compile(src, sys.argv[1], "exec")
    print("COMPILE OK")
except SyntaxError as e:
    print("SyntaxError at line", e.lineno, "col", e.offset)
    lines = src.split("\n")
    for i in range(max(0, e.lineno - 3), min(len(lines), e.lineno + 2)):
        print(i + 1, repr(lines[i]))

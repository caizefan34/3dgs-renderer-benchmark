import ast, sys
src=open("/mnt/storage_pool/liaoyuanjun/cgt_paper13/src/paper13_cgt.py").read()
try:
    ast.parse(src); print("SYNTAX OK")
except SyntaxError as e:
    print("SYNTAX ERROR:", e)
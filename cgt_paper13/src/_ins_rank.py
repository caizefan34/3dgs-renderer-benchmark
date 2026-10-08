f="/mnt/storage_pool/liaoyuanjun/cgt_paper13/src/paper13_cgt.py"
t=open(f).read()
anchor='    ap.add_argument("--seed", type=int, default=SEED)'
if "--rank" not in t and anchor in t:
    t=t.replace(anchor, anchor+"\n    ap.add_argument(\"--rank\", type=int, default=4)")
    open(f,"w").write(t); print("INSERTED --rank")
else:
    print("already present or anchor missing; rank in t:", "--rank" in t, "anchor in t:", anchor in t)
import ast; ast.parse(open(f).read()); print("SYNTAX OK")
import subprocess
print(subprocess.run(["grep","-n","--rank",f],capture_output=True,text=True).stdout)
import sys
s=open("_ssh_run.py").read()
s=s.replace('import subprocess, sys','import subprocess, sys\nsys.stdout.reconfigure(encoding="utf-8", errors="replace")\nsys.stderr.reconfigure(encoding="utf-8", errors="replace")')
open("_ssh_run.py","w").write(s)
print("ok")

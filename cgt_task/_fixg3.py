import re,io
p="/home/liaoyuanjun/cgt_realscene/final/SCRIPTS/g3_gpu_microbench.py"
s=open(p).read()
old="    dx = px[:, None, :] - mu[None, :, :]              # (P,M,2)"
new="    dx = torch.cat([px[:, None, :], py[:, None, :]], dim=2) - mu[None, :, :]   # (P,M,2)"
if new in s:
    print("ALREADY_FIXED")
elif old in s:
    s=s.replace(old,new,1)
    open(p,"w").write(s)
    print("PATCHED")
else:
    print("PATTERN_NOT_FOUND")
# show context
for i,l in enumerate(open(p).read().splitlines()[74:80],start=75):
    print(i,l)

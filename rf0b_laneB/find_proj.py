p = "rf0b_fastgs_model.py"
lines = open(p, encoding="utf-8").read().split("\n")
# find the K projection lines
for i,l in enumerate(lines):
    if "Xh2 = torch.cat" in l or "P = (K @" in l or "Xh2 = torch" in l:
        print(i+1, repr(l))
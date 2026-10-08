p = "rf0b_fastgs_model.py"
s = open(p, encoding="utf-8").read()
old = '''    l1 = torch.abs(render - gt).mean(0, keepdim=True)            # [1,H,W]
    l1 = l1 - l1.min()
    denom = (l1.max() - l1.min())
    l1_norm = l1 / denom if float(denom) > 1e-12 else torch.zeros_like(l1)
    return (l1_norm[0] > loss_thresh).float()                     # [H,W]'''
new = '''    # C0 renderer yields HWC [H,W,3]; reduce over the LAST (channel) axis.
    l1 = torch.abs(render - gt).mean(-1)                          # [H,W]
    l1 = l1 - l1.min()
    denom = (l1.max() - l1.min())
    l1_norm = l1 / denom if float(denom) > 1e-12 else torch.zeros_like(l1)
    return (l1_norm > loss_thresh).float()                         # [H,W]'''
assert old in s, "ndc block not found"
s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("NDC AXIS FIX APPLIED")
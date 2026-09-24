# -*- coding: utf-8 -*-
import glob
for f in glob.glob("scripts/phase-c29/[bdgk]*.py"):
    c = open(f).read()
    c = c.replace(").abs().mean()+0.2*(1.0-min(1.0,1.0)).backward()", ").abs().mean()+0.2*(1.0-min(1.0,1.0))\n    loss.backward()")
    c = c.replace("for p in model.parameters(): p.grad=None; opt.zero_grad(set_to_none=True); back = loss; exec('back.backward()'); ", "")
    open(f, "w").write(c)
    print("fixed:", f.split("/")[-1])

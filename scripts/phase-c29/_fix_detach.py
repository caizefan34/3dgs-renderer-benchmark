# -*- coding: utf-8 -*-
import glob
for f in glob.glob("scripts/phase-c29/[abdeghk]*.py"):
    c = open(f).read()
    c = c.replace('gt=m.render(cam).unsqueeze(0)', 'gt=m.render(cam).detach().unsqueeze(0)')
    open(f, 'w').write(c)
    print('fixed:', f.split('/')[-1])

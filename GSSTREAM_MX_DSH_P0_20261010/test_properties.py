#!/usr/bin/env python3
"""Property tests: no false negatives in conservative row intervals, CPU only."""
import math
import numpy as np
from cpu_oracle import xy_bounds,alpha,CUTOFF

rng=np.random.default_rng(20261010)
trials=400
counterexamples=[]
for t in range(trials):
    H=int(rng.integers(5,45));W=int(rng.integers(5,45))
    mu=rng.uniform(-10,max(H,W)+10,(1,2))
    major=math.exp(rng.uniform(math.log(.12),math.log(90)))
    minor=math.exp(rng.uniform(math.log(.12),math.log(7)))
    th=rng.uniform(-math.pi,math.pi)
    co=math.cos(th);si=math.sin(th)
    a=co*co/(major*major)+si*si/(minor*minor)
    b=co*si*(1/(major*major)-1/(minor*minor))
    c=si*si/(major*major)+co*co/(minor*minor)
    P=np.array([[a,b,c]],dtype=float)
    o=np.array([rng.uniform(0.002,0.999)])
    intervals=xy_bounds(0,mu,P,o,H,W,0)
    included={(y,x) for y,x0,x1 in intervals for x in range(x0,x1+1)}
    for y in range(H):
        for x in range(W):
            al,raw,_,_=alpha(0,x,y,mu,P,o)
            if al>=CUTOFF and (y,x) not in included:
                counterexamples.append(dict(trial=t,x=x,y=y,alpha=al))
                break
        if counterexamples: break
    if counterexamples:break
assert not counterexamples, f'unsafe culling! {counterexamples[:1]}'
print(f'PASS: {trials} random SPD/opacity/anisotropy tests, no valid pixel omitted (float64).')

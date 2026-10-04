"""Small FP64 BES math reference. All numerical callers must be guarded on MX.
Not a production bounds certificate or performance baseline.
"""
from __future__ import annotations
import bisect
import math


def rectangle_q_range(mu, conic, rect):
    """Analytic real-arithmetic extrema, evaluated in Python floats.
    rect=(xlo,xhi,ylo,yhi), conic=(a,b,c) for [[a,b],[b,c]].
    Does NOT certify a native floating-point exp/kernel.
    """
    mx, my = mu
    a, b, c = conic
    xl, xh, yl, yh = rect
    if a <= 0 or c <= 0 or a*c-b*b <= 0 or xl > xh or yl > yh:
        raise ValueError('SPD conic and ordered rectangle required')
    def q(x,y):
        dx,dy=x-mx,y-my
        return a*dx*dx+2*b*dx*dy+c*dy*dy
    corners=[q(x,y) for x in (xl,xh) for y in (yl,yh)]
    candidates=list(corners)
    if xl <= mx <= xh and yl <= my <= yh:
        candidates.append(0.0)
    for x in (xl,xh):
        y=min(yh,max(yl,my-b/c*(x-mx)))
        candidates.append(q(x,y))
    for y in (yl,yh):
        x=min(xh,max(xl,mx-b/a*(y-my)))
        candidates.append(q(x,y))
    return min(candidates),max(corners)


def effective_alpha(q, opacity, cap=0.99, cutoff=1/255):
    value=min(cap,opacity*math.exp(-0.5*q))
    return value if value >= cutoff else 0.0


def analytic_alpha_range(mu,conic,rect,opacity,cap=0.99,cutoff=1/255,
                         support_covers_rectangle=True):
    qlo,qhi=rectangle_q_range(mu,conic,rect)
    lo=effective_alpha(qhi,opacity,cap,cutoff)
    hi=effective_alpha(qlo,opacity,cap,cutoff)
    if not support_covers_rectangle:
        lo=0.0
    return lo,hi


def build_prefix(upper):
    prefix=[0.0]
    for u in upper:
        if not 0 < u < 1:
            raise ValueError('R upper bound must be strictly between zero and one')
        nxt=prefix[-1]-math.log1p(-u)
        if not math.isfinite(nxt) or nxt <= prefix[-1]:
            raise ValueError('unresolved prefix increment: route occurrence D')
        prefix.append(nxt)
    return prefix


def next_hit(alpha,upper,prefix,rng,start=0):
    """Return R ordinal (None=background) and count of evaluated candidates.
    alpha/upper are detached scalars; rejection skips the current occurrence.
    """
    if len(alpha)!=len(upper) or len(prefix)!=len(alpha)+1:
        raise ValueError('inconsistent list lengths')
    if not 0 <= start <= len(alpha):
        raise ValueError('invalid suffix')
    trials=0
    while start < len(alpha):
        uniform=rng.random()
        while uniform == 0:
            uniform=rng.random()
        z=prefix[start]-math.log(uniform)
        candidate=bisect.bisect_right(prefix,z,lo=start+1)-1
        if candidate >= len(alpha):
            return None,trials
        a,u=alpha[candidate],upper[candidate]
        if not 0 <= a <= u:
            raise ValueError('alpha violates upper bound')
        trials+=1
        if rng.random() < a/u:
            return candidate,trials
        start=candidate+1
    return None,trials


def full_image(alpha,colors,bg):
    """Torch tensors supplied by guarded caller, complete front-to-back blend."""
    value=bg*0
    trans=alpha.sum()*0+1
    for i in range(len(alpha)):
        value=value+trans*alpha[i]*colors[i]
        trans=trans*(1-alpha[i])
    return value+trans*bg


def conditional_image(alpha,colors,random_mask,hit,bg):
    value=bg*0
    trans=alpha.sum()*0+1
    end=len(alpha) if hit is None else hit
    for i in range(end):
        if not random_mask[i]:
            value=value+trans*alpha[i]*colors[i]
            trans=trans*(1-alpha[i])
    return value+trans*(bg if hit is None else colors[hit])


def hit_distribution(alpha,random_mask,after=None):
    """Exact differentiable distribution, global indices, including background."""
    trans=alpha.sum()*0+1
    outcomes=[]
    start=0 if after is None else after+1
    for i in range(start,len(alpha)):
        if random_mask[i]:
            outcomes.append((i,trans*alpha[i]))
            trans=trans*(1-alpha[i])
    outcomes.append((None,trans))
    return outcomes


def surrogate(alpha,colors,mask,hit,next_index,bg,residual):
    fh=conditional_image(alpha,colors,mask,hit,bg)
    loss=(residual.detach()*fh).sum()
    if hit is not None:
        fj=conditional_image(alpha,colors,mask,next_index,bg)
        score=(residual.detach()*(fh-fj)).sum().detach()
        loss=loss+score*alpha[hit].log()
    return loss


def enumerated_surrogate(alpha,colors,mask,bg,target):
    """Explicit expectation over independent A and B, plus B suffix J.
    All sampling probabilities are detached; AD only follows the estimator.
    """
    total=alpha.sum()*0
    outcomes=hit_distribution(alpha,mask)
    for ha,pa in outcomes:
        residual=(conditional_image(alpha,colors,mask,ha,bg)-target).detach()
        for hb,pb in outcomes:
            tails=[(None,alpha.sum()*0+1)] if hb is None else hit_distribution(alpha,mask,after=hb)
            for jb,pj in tails:
                weight=(pa*pb*pj).detach()
                total=total+weight*surrogate(alpha,colors,mask,hb,jb,bg,residual)
    return total

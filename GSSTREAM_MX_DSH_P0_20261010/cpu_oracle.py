#!/usr/bin/env python3
"""CPU-only experimental oracle for conservative ellipse row support.
Not a GPU speedup implementation, and NOT a novelty claim.
Only supports projected classic RGB/no-other-loss synthetic or exported snapshots.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

CUTOFF=1./255.
TERMINATION=1.e-4


def validate(means, conics, opa, colors, depth, H, W):
    n=len(opa)
    assert 0 < n <= 100000 and 1 <= H <= 4096 and 1 <= W <= 4096
    assert means.shape==(n,2) and conics.shape==(n,3) and colors.shape==(n,3) and depth.shape==(n,)
    assert np.isfinite(means).all() and np.isfinite(conics).all()
    assert np.isfinite(opa).all() and np.isfinite(colors).all() and np.isfinite(depth).all()
    assert np.logical_and(opa>=0,opa<=1).all()
    a,b,c=conics.T
    assert (a>0).all() and (c>0).all() and (a*c-b*b>0).all(), 'SPD invalid; in production fall back to dense, never cull'


def inputs(args):
    if args.npz:
        with np.load(args.npz,allow_pickle=False) as z:
            for k in ('means2d','conics','opacities','colors','depths','height','width'):
                if k not in z: raise ValueError('Missing key '+k)
            mu=np.asarray(z['means2d'],dtype=np.float64)
            P=np.asarray(z['conics'],dtype=np.float64)
            o=np.asarray(z['opacities'],dtype=np.float64).reshape(-1)
            c=np.asarray(z['colors'],dtype=np.float64)
            d=np.asarray(z['depths'],dtype=np.float64).reshape(-1)
            H=int(z['height']);W=int(z['width'])
        source='npz:'+str(args.npz)
    else:
        g=np.random.default_rng(args.seed)
        H=W=args.size
        n=args.num
        mu=g.uniform(-.3*W,1.3*W,(n,2))
        sig0=np.exp(g.uniform(np.log(.2), np.log(max(W,10)*1.3),n))
        sig1=np.exp(g.uniform(np.log(.2), np.log(max(W,10)*.6),n))
        ang=g.uniform(-np.pi,np.pi,n)
        cs=np.cos(ang);ss=np.sin(ang)
        a=(cs/sig0)**2+(ss/sig1)**2
        b=cs*ss*(1/sig0**2-1/sig1**2)
        cc=(ss/sig0)**2+(cs/sig1)**2
        P=np.stack((a,b,cc),axis=1)
        o=g.uniform(.004, .98,n)
        c=g.uniform(0.,1.,(n,3))
        d=g.uniform(.1,100,n)
        # Explicit high-opacity foreground + wide anisotropic background stress cases
        if n>2:
            mu[0]=[W/2,H/2];P[0]=[1/10000,0,1/10000];o[0]=.9;d[0]=100
            mu[1]=[W/2,H/2];P[1]=[1/.4**2,0,1/.4**2];o[1]=.999;d[1]=.1
            mu[2]=[W/2,H/2];P[2]=[1/.25**2,0,1/30**2];o[2]=.9;d[2]=.3
        source='synthetic'
    validate(mu,P,o,c,d,H,W)
    return mu,P,o,c,d,H,W,source


def xy_bounds(i,mu,P,o,H,W,pad):
    if o[i]<CUTOFF: return []
    a,b,c=P[i]
    det=a*c-b*b
    tau=2*math.log(o[i]/CUTOFF)
    # Conservative ymax: projection of ellipse onto y axis.
    maxy=math.sqrt(tau*a/det)
    y0=max(0, int(math.floor(mu[i,1]-maxy-pad-2)))
    y1=min(H-1, int(math.ceil(mu[i,1]+maxy+pad+2)))
    result=[]
    for y in range(y0,y1+1):
        dy=y-mu[i,1]
        discr=a*tau-det*dy*dy
        # + numerical slack when near boundary
        discr=max(0.,discr)
        if discr==0 and abs(dy)>maxy+1e-10: continue
        dx1=(-b*dy-math.sqrt(discr))/a
        dx2=(-b*dy+math.sqrt(discr))/a
        # Wider interval is intentionally conservative; exact cutoff rechecked per pixel.
        x0=max(0,math.floor(mu[i,0]+dx1-pad-2))
        x1=min(W-1,math.ceil(mu[i,0]+dx2+pad+2))
        if x0<=x1: result.append((y,int(x0),int(x1)))
    return result


def tile_bbox_count(mu,P,o,H,W,ts):
    """Gaussian bbox/TILE visits of CPU baseline proxy (not actual CUDA steps)."""
    cost=0
    tiles=0
    for i in range(len(o)):
        if o[i]<CUTOFF:continue
        a,b,c=P[i];det=a*c-b*b;tau=2*math.log(o[i]/CUTOFF)
        maxx=math.sqrt(tau*c/det); maxy=math.sqrt(tau*a/det)
        lx=max(0,int(math.floor((mu[i,0]-maxx-2)/ts)))
        rx=min((W-1)//ts,int(math.floor((mu[i,0]+maxx+2)/ts)))
        ly=max(0,int(math.floor((mu[i,1]-maxy-2)/ts)))
        ry=min((H-1)//ts,int(math.floor((mu[i,1]+maxy+2)/ts)))
        if lx>rx or ly>ry:continue
        for ty in range(ly,ry+1):
            for tx in range(lx,rx+1):
                cost+=min(ts,W-tx*ts)*min(ts,H-ty*ts)
                tiles+=1
    return cost,tiles


def alpha(g,x,y,mu,P,o):
    dx=x-mu[g,0];dy=y-mu[g,1]
    a,b,c=P[g]
    sigma=.5*(a*dx*dx+2*b*dx*dy+c*dy*dy)
    raw=o[g]*math.exp(-max(0.,sigma))
    return min(.999,raw),raw,dx,dy


def evaluate(mu,P,o,colors,depth,H,W,ordered,pad=0):
    """Independent CPU dense vs conservative-row oracle, same depth order.
       Calculates RGB, alpha, and analytic gradients of mean2d/opacity/RGB.
       Uses a deterministic toy RGB MSE objective and 1e-4 transmittance stop.
    """
    n=len(o)
    rank=np.argsort(depth,kind='stable')
    if ordered:
        events=[[] for _ in range(H*W)]
        intervals=0; candidates=0
        for ix in rank:
            for y,x0,x1 in xy_bounds(ix,mu,P,o,H,W,pad):
                intervals+=1;candidates+=x1-x0+1
                for x in range(x0,x1+1):events[y*W+x].append(int(ix))
    else:
        events=None; intervals=0;candidates=H*W*n
    target=np.full((H,W,3),.4,dtype=np.float64)
    bg=np.array([.07,.12,.2],dtype=np.float64)
    render=np.empty((H,W,3),dtype=np.float64)
    gcol=np.zeros_like(colors);gopa=np.zeros_like(o);gmean=np.zeros_like(mu)
    hits=0;visited=0;truncated=0
    for y in range(H):
        for x in range(W):
            acc=np.zeros(3,dtype=np.float64);T=1.
            rec=[]
            stream=events[y*W+x] if ordered else rank
            for ix in stream:
                visited+=1
                al,raw,dx,dy=alpha(ix,x,y,mu,P,o)
                if al<CUTOFF:continue
                rec.append((ix,al,raw,dx,dy,T))
                acc+=T*al*colors[ix]
                T*=1.-al
                hits+=1
                if T<TERMINATION:
                    truncated+=1
                    break
            final=acc+T*bg
            render[y,x]=final
            q=final-target[y,x]
            adjT=float(q@bg)
            for ix,al,raw,dx,dy,preT in reversed(rec):
                gcol[ix]+=q*(preT*al)
                adjA=float(q@(preT*colors[ix])-adjT*preT)
                adjT=float(q@(al*colors[ix])+adjT*(1.-al))
                if raw<.999:
                    expval=raw/o[ix] if o[ix]>0 else 0
                    gopa[ix]+=adjA*expval
                    a,b,c=P[ix]
                    gmean[ix]+=adjA*raw*np.array([a*dx+b*dy,b*dx+c*dy])
    return dict(rgb=render,gcol=gcol,gopa=gopa,gmean=gmean,hits=hits,visited=visited,
                intervals=intervals,candidates=candidates,truncated=truncated)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--npz',type=Path,help='Existing CPU-exported frozen projected arrays only; NEVER invoke GPU')
    ap.add_argument('--output',type=Path,default=Path('p0_cpu_result'))
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--size',type=int,default=40)
    ap.add_argument('--num',type=int,default=75)
    ap.add_argument('--tile-size',type=int,default=16)
    args=ap.parse_args()
    mu,P,o,c,d,H,W,source=inputs(args)
    if H*W*len(o)>10_000_000:raise ValueError('CPU preflight work cap 10M dense visits; sample fixed cameras/Gaussians without GPU')
    dense=evaluate(mu,P,o,c,d,H,W,False)
    sparse=evaluate(mu,P,o,c,d,H,W,True)
    err={k:float(np.max(np.abs(dense[k]-sparse[k]))) for k in ('rgb','gcol','gopa','gmean')}
    work={}
    for ts in sorted(set([8,16,32,args.tile_size])):
        visit,tile=tile_bbox_count(mu,P,o,H,W,ts)
        work[str(ts)]={'bbox_candidate_pixel_visits':int(visit),'tile_intersections':int(tile),
                       'ratio_active_hits_to_candidate_visits':float(sparse['hits']/max(visit,1))}
    result={
        'source':source,'H':H,'W':W,'N':len(o),'cutoff':CUTOFF,'termination':TERMINATION,
        'toy_objective':'half-sum-squared-RGB residual, no SSIM/3D projection; NOT native training parity',
        'forward_backward_errors':err,'exact_visit_count':dense['hits'],'stream_visit_count':sparse['hits'],
        'row_intervals':sparse['intervals'],'stream_candidates_including_conservative_padding':sparse['candidates'],
        'cost_proxies':work,
        'status':'CPU_BASIC_PARITY_PASS' if max(err.values())<1e-10 and dense['hits']==sparse['hits'] else 'FAIL_STOP',
        'limitations':['CPU double precision toy only','GPU float cutoff/clamp and sort semantics not verified',
        'real scene 1080p projection, loss, SH, absgrad and early termination not verified',
        'interval generation, CSR materialization, sort, memory traffic and GPU execution overhead unmeasured',
        'No novelty claim; Speedy-Splat AccuTile and gsplat rasterize_to_indices_in_range are nearest prior art']
    }
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'cpu_oracle.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    digest=hashlib.sha256((args.output/'cpu_oracle.json').read_bytes()).hexdigest()
    (args.output/'SHA256SUMS').write_text(digest+'  cpu_oracle.json\n',encoding='ascii')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if result['status']!='CPU_BASIC_PARITY_PASS':raise SystemExit(3)

if __name__=='__main__':main()

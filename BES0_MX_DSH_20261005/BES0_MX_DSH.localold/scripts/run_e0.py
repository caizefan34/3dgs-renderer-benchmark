#!/usr/bin/env python3
"""Run only through the MX launcher. CPU numerical work is still MX-only."""
from mx_guard import require_guarded_process


def main():
    cfg,proof=require_guarded_process()  # before scientific imports/computation
    import hashlib
    import json
    import math
    import os
    from pathlib import Path
    import random
    import torch
    import bes_reference as ref
    from provenance import protocol_digest,implementation_digest,reference_digest

    if not torch.__version__.startswith(cfg['torch_version']):
        raise RuntimeError('BLOCKED_ENV: unexpected torch version')
    torch.set_num_threads(1)
    output=Path(os.environ['BES0_OUTPUT_DIR'])
    checks={}
    results={}
    masks=[(False,False,False,False),(True,True,True,True),
           (True,False,True,False),(False,True,False,True),
           (False,False,True,False)]
    color_errors=[]; gradient_errors=[]
    bg=torch.tensor([0.07,0.12,0.23],dtype=torch.float64)
    target=torch.tensor([0.31,0.2,0.63],dtype=torch.float64)
    for fixture in ('direct_alpha','geometry_chain'):
        for mask in masks:
            param=torch.tensor([[-1.7,0.2,-0.3,0.4],[-0.2,-0.1,0.5,-0.2],
                                [0.9,0.4,0.1,0.7],[-2.8,-0.4,-0.2,0.0]],
                               dtype=torch.float64,requires_grad=True)
            colors=torch.tensor([[0.2,0.3,0.8],[0.8,0.6,0.1],
                                 [0.1,0.9,0.4],[0.7,0.2,0.6]],
                                dtype=torch.float64,requires_grad=True)
            alpha=param[:,0].sigmoid()
            viewed_colors=colors
            if fixture=='geometry_chain':
                # Smooth opacity / mean / scale / color-direction dependency.
                # Does not assert actual native SH chain or cutoff derivative.
                q=((param[:,1]-.13)**2+(param[:,2]+.17)**2)/param[:,3].exp()
                alpha=alpha*torch.exp(-.5*q)
                viewed_colors=colors+.03*param[:,1:2]
            native=ref.full_image(alpha,viewed_colors,bg)
            expectation=native*0
            for h,p in ref.hit_distribution(alpha,mask):
                expectation=expectation+p*ref.conditional_image(alpha,viewed_colors,mask,h,bg)
            ce=(expectation-native).abs().max().item()
            loss=.5*((native-target)**2).sum()
            g0=torch.autograd.grad(loss,(param,colors),retain_graph=True)
            estimator=ref.enumerated_surrogate(alpha,viewed_colors,mask,bg,target)
            g1=torch.autograd.grad(estimator,(param,colors))
            ge=max((a-b).abs().max().item() for a,b in zip(g0,g1))
            color_errors.append(ce); gradient_errors.append(ge)
    checks['color_expectation']=max(color_errors)<=1e-10
    checks['gradient_expectation']=max(gradient_errors)<=1e-9
    results['identities']={'fixtures':10,'color_maxabs':max(color_errors),
                           'gradient_maxabs':max(gradient_errors),
                           'parameter_scope':'alpha/color and smooth geometry-chain; native CUDA not tested'}

    # Continuous rectangle formula checked on a dense grid only as a reference
    # regression, NOT as a floating-point production certificate.
    rectangles=[((.2,.3),(2.,.4,1.5),(-1.,1.,-1.,1.)),
                ((2.,-.4),(1.,-.3,3.),(-.5,.5,-.8,.8)),
                ((-2.,2.),(3.,1.,2.),(0.,1.,0.,1.))]
    bound_ok=True
    for mu,A,rect in rectangles:
        lo,hi=ref.rectangle_q_range(mu,A,rect)
        for xi in range(31):
            for yi in range(31):
                x=rect[0]+(rect[1]-rect[0])*xi/30
                y=rect[2]+(rect[3]-rect[2])*yi/30
                dx,dy=x-mu[0],y-mu[1]
                q=A[0]*dx*dx+2*A[1]*dx*dy+A[2]*dy*dy
                bound_ok &= lo-1e-12 <= q <= hi+1e-12
    lo,hi=ref.analytic_alpha_range((0.,0.),(1.,0.,1.),(-1.,1.,-1.,1.),.5,
                                 support_covers_rectangle=False)
    checks['analytic_bounds_reference']=bound_ok and lo==0 and hi>0

    alpha=[.03,.2,.08,.4,.015]
    upper=[.055,.35,.12,.65,.027]
    prefix=ref.build_prefix(upper)
    samples=120000
    rng=random.Random(19027)
    counts=[0]*(len(alpha)+1)
    trial_hist=[0]*(len(alpha)+1)
    trans=1.; expected=[]; q_expected=0.
    for a,u in zip(alpha,upper):
        expected.append(trans*a); q_expected+=trans*u; trans*=1-a
    expected.append(trans)
    qsum=0; qsum2=0
    for _ in range(samples):
        h,q=ref.next_hit(alpha,upper,prefix,rng)
        counts[len(alpha) if h is None else h]+=1
        trial_hist[q]+=1; qsum+=q; qsum2+=q*q
    zscores=[abs(n/samples-p)/math.sqrt(p*(1-p)/samples) for n,p in zip(counts,expected)]
    mean=qsum/samples
    se=math.sqrt(max(0.,qsum2/samples-mean*mean)/samples)
    checks['first_hit_distribution']=max(zscores)<=6
    checks['candidate_count_identity']=abs(mean-q_expected)<=6*se+1e-6
    checks['candidate_count_bound']=q_expected<=2 and mean<=2+6*se
    tail_checks=[]
    for k in range(1,6):
        observed=sum(trial_hist[k+1:])/samples
        bound=(.5)**k
        margin=6*math.sqrt(bound*(1-bound)/samples)+6/samples
        tail_checks.append(observed<=bound+margin)
    checks['trial_tail_bound']=all(tail_checks)
    results['sampling']={'samples':samples,'seed':19027,'alpha':alpha,'upper':upper,
                         'observed_counts':counts,'exact_probabilities':expected,
                         'max_zscore':max(zscores),'trial_histogram':trial_hist,
                         'mean_trials':mean,'exact_mean_trials':q_expected,
                         'standard_error_trials':se,'gamma':2}
    # Suffix query after first event is a different distribution.
    rng=random.Random(19028)
    suffix_counts=[0]*(len(alpha)+1)
    suffix_expected=[0.]*(len(alpha)+1); t=1.
    for i in range(2,len(alpha)):
        suffix_expected[i]=t*alpha[i]; t*=1-alpha[i]
    suffix_expected[-1]=t
    for _ in range(60000):
        h,q=ref.next_hit(alpha,upper,prefix,rng,start=2)
        suffix_counts[len(alpha) if h is None else h]+=1
    checks['suffix_distribution']=all(
        abs(n/60000-p)<=6*math.sqrt(p*(1-p)/60000)+1e-6
        for n,p in zip(suffix_counts,suffix_expected))
    results['suffix']={'counts':suffix_counts,'probabilities':suffix_expected,'samples':60000}

    a,u=.2,.6
    wrong=u*(-math.log1p(-a))/(-math.log1p(-u))
    checks['wrong_hazard_ratio_rejected']=abs(wrong-a)>.01
    checks['empty_random_set']=ref.next_hit([],[],[0.],random.Random(1))==(None,0)
    checks['certain_background']=ref.next_hit([0.],[.3],ref.build_prefix([.3]),random.Random(1))[0] is None
    results['wrong_acceptance_counterexample']={'target':a,'wrong_probability':wrong}
    detail=output/'E0_DETAILS.json'
    detail.write_text(json.dumps(results,indent=2)+'\n')
    stage={'schema':'bes0.stage.v1','stage':'E0','status':'PASS' if all(checks.values()) else 'FAIL',
           'protocol_sha256':protocol_digest(),'implementation_sha256':implementation_digest(),
           'reference_sha256':reference_digest(),
           'source_commit':None,'guard_sha256':hashlib.sha256(Path(os.environ['BES0_GUARD_PROOF']).read_bytes()).hexdigest(),
           'selected_config_sha256':None,'inputs_verified':True,'checks':checks,
           'color_maxabs':max(color_errors),'gradient_maxabs':max(gradient_errors),
           'raw_files':[{'path':'E0_DETAILS.json','sha256':hashlib.sha256(detail.read_bytes()).hexdigest()}],
           'scope':'MX mathematical reference only; no production native certificate',
           'guard':proof['hostname']}
    (output/'E0.json').write_text(json.dumps(stage,indent=2)+'\n')
    print(json.dumps(stage,indent=2))
    return 0 if stage['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())

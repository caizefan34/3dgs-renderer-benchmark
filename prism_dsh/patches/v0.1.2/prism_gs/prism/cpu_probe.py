"""A deterministic image-level smoke experiment; CPU timings are not a speed claim."""
import argparse,json,time
import torch
from .geometry import project,fit_consensus,unpack_geometry
from .oracle import fixture,render,loss


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='results/cpu_probe.json');args=ap.parse_args()
    torch.set_num_threads(1)
    truth,v,k,colors,opacity,w,h=fixture()
    x=truth.clone();x[:,:3]+=torch.tensor([.025,-.015,.035]);x[:,3]+=.025
    tz,td=project(truth,v,k,w,h)
    targets=[render(tz[i],td[i],colors,opacity,h,w).detach() for i in range(5)]
    def evaluate(xx):
        z,d=project(xx,v,k,w,h)
        return [float(loss(render(z[i],d[i],colors,opacity,h,w),targets[i])) for i in range(5)]
    before=evaluate(x);z0,dep=project(x,v[:3],k[:3],w,h)
    z=z0.detach().clone().requires_grad_();opt=torch.optim.Adam([z],lr=.03)
    scale=torch.tensor([1,1,.1,.1,.1],dtype=x.dtype)
    start=time.perf_counter();local=[]
    for _ in range(3):
        opt.zero_grad()
        photo=sum(loss(render(z[i],dep[i],colors,opacity,h,w),targets[i]) for i in range(3))/3
        objective=photo+.001*((z-z0)/scale).square().mean()
        objective.backward();opt.step()
        with torch.no_grad():z.copy_(z0+(z-z0).clamp(-scale,scale))
        local.append(float(photo.detach()))
    fitted,history=fit_consensus(x,z.detach(),v[:3],k[:3],w,h,
        torch.ones_like(z)/scale.square(),steps=5)
    elapsed=time.perf_counter()-start
    after=evaluate(fitted)
    # Matched full-render exposure control in the SAME covariance coordinates.
    # This does not stand in for the production optimizer or a tuned LM baseline.
    direct=x.clone().requires_grad_();opt2=torch.optim.Adam([direct],lr=.003)
    for _ in range(3):
        opt2.zero_grad();zz,dd=project(direct,v[:3],k[:3],w,h)
        objective=sum(loss(render(zz[i],dd[i],colors,opacity,h,w),targets[i]) for i in range(3))/3
        objective.backward();opt2.step()
    control=evaluate(direct.detach())
    result={'status':'CPU_SMOKE_ONLY','gpu_tested':False,'speedup_claim':None,
        'limitations':['tiny full-support oracle; not gsplat','fixed colors/opacity',
                      'no structure changes','no primal-dual iteration; one proximal proposal',
                      'control LR is specified, not tuned; no superiority verdict'],
        'before_per_view_loss':before,'lifted_local_photo_loss':local,
        'after_consensus_per_view_loss':after,'direct_adam_per_view_loss':control,
        'fit_history':history,'proposal_cpu_seconds':elapsed,
        'train_views':[0,1,2],'audit_views':[3,4],
        'proposal_train_delta':sum(after[:3])/3-sum(before[:3])/3,
        'proposal_audit_delta':sum(after[3:])/2-sum(before[3:])/2,
        'render_backward_views_per_arm':9}
    with open(args.out,'w') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

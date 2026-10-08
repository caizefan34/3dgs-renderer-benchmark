"""Run PRISM's first real-scene probe using the frozen EPIC-05 bootstrap.

Example (from this bundle root):
  python -m prism.run_gpu --gx-path /root/epic05_grc3d/remote_scripts \
    --gpu 0 --scene bicycle --snapshot /path/snap_bicycle_005000.pt \
    --camera-seq /path/camera_sequence.npy --out results/bicycle_5k.json

This is a fixed-structure GEOMETRY-ONLY probe, not a full training benchmark.
No split/clone/prune, persistent dual, speedup estimate or automatic GO verdict.
"""
import argparse,hashlib,importlib,json,os,platform,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from . import gsplat_backend as B
from .geometry import covariance,pack_geometry,unpack_geometry,project,fit_consensus,jacobian


def save(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(p)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def sync_time():
    torch.cuda.synchronize();return time.perf_counter()


def grad_error(a,b):
    diff=(a-b).double()
    return {'maxabs':float(diff.abs().max()),
        'relative_l2':float(diff.norm()/a.double().norm().clamp_min(1e-20))}


def m0(params,cam,gt,degree,loss):
    pp={k:v.detach().clone().requires_grad_() for k,v in params.items()}
    original,st=B.state(pp,cam,degree)
    lower=B.draw(st)
    ga=torch.autograd.grad(loss(original,gt),tuple(pp.values()),retain_graph=True,allow_unused=True)
    gb=torch.autograd.grad(loss(lower,gt),tuple(pp.values()),allow_unused=True)
    ga=[torch.zeros_like(p) if g is None else g for p,g in zip(pp.values(),ga)]
    gb=[torch.zeros_like(p) if g is None else g for p,g in zip(pp.values(),gb)]
    errs={k:grad_error(a,b) for k,a,b in zip(pp,ga,gb)}
    value_error=float((original-lower).abs().max())
    with torch.no_grad():
        st=B.detach_state(st)
        valid=st['mask'][0].nonzero().flatten()[:256]
        if len(valid)==0:raise RuntimeError('M0 camera has no visible Gaussians')
        zero=B.draw(st,valid,B.features(st,valid))
        zero_error=float((original-zero).abs().max())
        raw_repeat=B.baseline(params,cam,degree)[0]
        repeat_error=float((original-raw_repeat).abs().max())
        cov=covariance(params['quats'],params['log_scales'])
        cov_image=B.baseline(params,cam,degree,covars=cov)[0]
        cov_error=float((original-cov_image).abs().max())
        x=pack_geometry(params['means'][valid],cov[valid])
        pure,_=project(x.double(),cam['viewmat'][None].double(),cam['K'][None].double(),cam['width'],cam['height'])
        ref=B.features(st,valid).double()
        proj_uv=float((pure[0,:,:2]-ref[:,:2]).abs().max())
        proj_shape=float((pure[0,:,2:]-ref[:,2:]).abs().max())
    # Fixed, predeclared tolerances; do not enlarge after seeing GPU noise.
    passed=(value_error<=2e-6 and zero_error<=2e-6 and cov_error<=2e-5 and
            proj_uv<=2e-3 and proj_shape<=2e-3 and
            all(v['relative_l2']<=1e-3 or v['maxabs']<=1e-8 for v in errs.values()))
    return dict(status='PASS_ADAPTER' if passed else 'BLOCKED_M0',
        forward_maxabs=value_error,zero_override_maxabs=zero_error,
        quaternion_to_cov_render_maxabs=cov_error,baseline_repeat_forward_maxabs=repeat_error,
        projection_uv_maxabs=proj_uv,projection_shape_maxabs=proj_shape,
        group_gradients=errs,trainer_independent_parity='REQUIRED_SEPARATELY',
        note='Passing this adapter check is not independent original-trainer parity.')


def main():
    ap=argparse.ArgumentParser()
    for name in ('gx-path','scene','snapshot','camera-seq','out'):ap.add_argument('--'+name,required=True)
    ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--max-active',type=int,default=512)
    ap.add_argument('--inner-steps',type=int,default=3)
    ap.add_argument('--seed',type=int,default=20260927)
    ap.add_argument('--m0-only',action='store_true')
    args=ap.parse_args()
    out={'status':'STARTED','gpu_speedup':None,'scope':'fixed-structure geometry-only one-proposal diagnostic',
         'args':vars(args),'limitations':['not full training','no persistent dual or structural changes',
         'Python reference timings are not optimized-kernel costs','no tuned LM control yet']}
    save(args.out,out)
    try:
        os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
        sys.path.insert(0,args.gx_path)
        G=importlib.import_module('gx_common')
        trainer,ident=G.boot_gmain(args.gpu)
        out['identity']=G.run_identity(trainer,ident,args.gpu)
        out['fused_loss']=G.check_fused_lock()
        out['input_sha256']={p:sha(p) for p in (args.snapshot,args.camera_seq)}
        out['adapter_source_sha256']=sha(B.__file__)
        out['gpu_before']=G.gpu_inventory()
        # Shared GPU occupancy is evidence only; no automatic speed conclusion.
        torch.manual_seed(args.seed)
        snap=torch.load(args.snapshot,map_location='cpu',weights_only=False)
        params={'means':snap['xyz'].cuda().float(),'quats':snap['rotation'].cuda().float(),
                'log_scales':snap['scaling'].cuda().float(),
                'opacity_logits':snap['opacity'].cuda().float().reshape(-1),
                'sh0':snap['shs'][:,:1].cuda().float(),'shN':snap['shs'][:,1:].cuda().float()}
        degree=int(snap['active_sh_degree'])
        sequence=np.load(args.camera_seq)
        camera_ids=[]
        for c in sequence:
            if int(c) not in camera_ids:camera_ids.append(int(c))
            if len(camera_ids)==5:break
        if len(camera_ids)<5:raise RuntimeError('Need 5 distinct TRAINING camera IDs')
        out['camera_rule']='first 5 distinct IDs in frozen TRAINING sequence; first 3 optimize, last 2 audit'
        out['camera_ids']=camera_ids
        dataset=G.build_dataset(args.scene)
        cameras=[];targets=[]
        for c in camera_ids:
            cam,gt=dataset.get_item(c);cam=G.camera_dict(cam)
            if tuple(gt.shape)==(3,cam['height'],cam['width']):gt=gt.permute(1,2,0)
            if tuple(gt.shape)!=(cam['height'],cam['width'],3):raise RuntimeError('Invalid target dimensions')
            cameras.append(cam);targets.append(gt.float().contiguous())
        if len({(c['width'],c['height']) for c in cameras[:3]})!=1:
            raise RuntimeError('Pilot requires equal training-view resolution; NEVER resize to hide this')
        out['m0']=[m0(params,cameras[i],targets[i],degree,G.fused_loss) for i in range(3)]
        save(args.out,out)
        if any(v['status']!='PASS_ADAPTER' for v in out['m0']):
            out['status']='BLOCKED_M0';save(args.out,out);return
        if args.m0_only:
            out['status']='M0_ADAPTER_PASS_ONLY';save(args.out,out);return
        pre_start=sync_time()
        with torch.no_grad():
            states=[];before=[]
            for c,t in zip(cameras,targets):
                image,st=B.state(params,c,degree);before.append(float(G.fused_loss(image,t)));states.append(B.detach_state(st))
            valid=torch.stack([s['mask'][0] for s in states[:3]]).all(0)
            eligible=valid.nonzero().flatten()
            if len(eligible)<8:raise RuntimeError('Insufficient common visibility: UNOBSERVABLE, not mechanism DROP')
            permutation=torch.randperm(len(eligible),device='cuda')
            ids=eligible[permutation[:args.max_active]]
            cov=covariance(params['quats'],params['log_scales'])
            pack_stats={}
            x0=pack_geometry(params['means'][ids],cov[ids],stats=pack_stats)
            z0=torch.stack([B.features(s,ids) for s in states[:3]])
        out['selection']={'N':len(params['means']),'common_visible':len(eligible),'active':len(ids),
            'active_fraction':len(ids)/len(params['means']),'ids':ids.cpu().tolist()}
        out['geometry_pack']=pack_stats
        out['before_per_view_loss']=before
        views=torch.stack([c['viewmat'] for c in cameras[:3]])
        ks=torch.stack([c['K'] for c in cameras[:3]])
        w,h=cameras[0]['width'],cameras[0]['height']
        scale=x0.new_tensor([1,1,.1,.1,.1])
        _,jcheck=jacobian(x0.double(),views.double(),ks.double(),w,h)
        pos=jcheck[:,:,:2,:3].permute(1,0,2,3).reshape(len(ids),6,3)
        singular=torch.linalg.svdvals(pos)
        relative=singular[:,-1]/singular[:,0].clamp_min(1e-20)
        out['position_observability']={'min_relative_singular':float(relative.min()),
            'median_relative_singular':float(relative.median()),
            'fraction_below_1e_minus_6':float((relative<1e-6).double().mean())}
        del jcheck,pos,singular
        out['shared_setup_seconds']=sync_time()-pre_start
        def actual(xx):
            means=params['means'].clone();covars=cov.clone()
            m,co=unpack_geometry(xx);means[ids]=m;covars[ids]=co
            return means,covars
        def evaluate(xx):
            means,covars=actual(xx)
            with torch.no_grad():return [float(G.fused_loss(B.baseline(params,c,degree,covars,means)[0],t))
                for c,t in zip(cameras,targets)]
        out['arms']={}
        # All arms start at the same state. 1-step arm is a diagnostic, not LM.
        for arm,steps in [('screen_one_step',1),('screen_nonlinear',args.inner_steps),('direct_geometry_adam',args.inner_steps)]:
            start=sync_time();torch.cuda.reset_peak_memory_stats();trace=[];fit=[]
            if arm.startswith('screen'):
                z=z0.detach().clone().requires_grad_();opt=torch.optim.Adam([z],lr=.03)
                for step in range(steps):
                    opt.zero_grad()
                    photo=sum(G.fused_loss(B.draw(states[i],ids,z[i]),targets[i]) for i in range(3))/3
                    objective=photo+.001*((z-z0)/scale).square().mean()
                    objective.backward();opt.step()
                    with torch.no_grad():z.copy_(z0+(z-z0).clamp(-scale,scale))
                    trace.append({'step':step,'photo_before_step':float(photo.detach())})
                # Float64 small solves avoid conflating ill-conditioning with FP32 noise.
                xx,fit=fit_consensus(x0.double(),z.detach().double(),views.double(),ks.double(),w,h,
                    torch.ones_like(z,dtype=torch.float64)/scale.double().square(),steps=3)
                xx=xx.float()
            else:
                xx=x0.detach().clone().requires_grad_();opt=torch.optim.Adam([xx],lr=.003)
                for step in range(steps):
                    opt.zero_grad();means,covars=actual(xx)
                    photo=sum(G.fused_loss(B.baseline(params,cameras[i],degree,covars,means)[0],targets[i])
                              for i in range(3))/3
                    photo.backward();opt.step()
                    trace.append({'step':step,'photo_before_step':float(photo.detach())})
                xx=xx.detach()
            optimization_seconds=sync_time()-start
            losses=evaluate(xx)
            elapsed=sync_time()-start
            out['arms'][arm]={'losses':losses,'trace':trace,'fit_history':fit,
                'optimization_seconds':optimization_seconds,'with_evaluation_seconds':elapsed,
                'full_resolution_backward_views':steps*3,'evaluation_forward_views':5,
                'shared_setup_must_be_added':True,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'train_delta':sum(losses[:3])/3-sum(before[:3])/3,
                'audit_delta':sum(losses[3:])/2-sum(before[3:])/2}
            save(args.out,out)
        out['gpu_after']=G.gpu_inventory();out['status']='PROBE_COMPLETE_NO_GO_VERDICT'
        out['required_next']='Original-trainer parity; multi-seed/paired order; tuned raw-parameter Adam and LM; equal-wall-time trajectories'
        save(args.out,out)
    except Exception:
        out['status']='ERROR';out['traceback']=traceback.format_exc();save(args.out,out);raise


if __name__=='__main__':main()

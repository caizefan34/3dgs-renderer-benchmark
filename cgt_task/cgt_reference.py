#!/usr/bin/env python3
"""CGT research reference, 2026-10-03.

Evidence scope: CPU algebra checks and a synthetic 2D ray-domain fit.
This is NOT a 3D renderer, a CUDA benchmark, or a training speedup result.
Requires Python 3.10+, numpy, scipy, matplotlib.
Run: OPENBLAS_NUM_THREADS=1 python3 cgt_reference.py --output ./results
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, softmax


@dataclass
class TransferMixture:
    precision: np.ndarray
    natural: np.ndarray
    bias: np.ndarray
    # Each row contains [premultiplied R, G, B, transmittance].
    expert: np.ndarray

    def evaluate(self, q, jacobian=False):
        logits = (-0.5 * np.einsum('nd,kde,ne->nk', q, self.precision, q)
                  + q @ self.natural.T + self.bias)
        weights = softmax(logits, axis=1)
        value = weights @ self.expert
        if not jacobian:
            return value
        score = self.natural[None] - np.einsum('kde,ne->nkd', self.precision, q)
        jac = np.einsum('nk,nkc,nkd->ncd', weights,
                        self.expert[None] - value[:, None], score)
        return value, jac

    def compose(self, behind):
        """Exact symbolic composition on a COMMON ray domain and fixed order.

        No geometric masks, changing intersection order, or recompression here.
        Component count multiplies. The operation is associative, not commutative.
        """
        a, b = self.expert, behind.expert
        e = a[:, None, :3] + a[:, None, 3:4] * b[None, :, :3]
        t = a[:, None, 3] * b[None, :, 3]
        out = np.concatenate([e, t[..., None]], axis=-1)
        k = len(a) * len(b)
        return TransferMixture(
            (self.precision[:, None] + behind.precision[None]).reshape(k, *self.precision.shape[1:]),
            (self.natural[:, None] + behind.natural[None]).reshape(k, -1),
            (self.bias[:, None] + behind.bias[None]).reshape(k),
            out.reshape(k, 4),
        )

    def restrict_affine(self, matrix, offset):
        """Exact restriction q = matrix @ u + offset, in a fixed ray chart."""
        precision = np.einsum('di,kde,ej->kij', matrix, self.precision, matrix)
        natural = (self.natural - np.einsum('kde,e->kd', self.precision, offset)) @ matrix
        bias = (self.bias + self.natural @ offset
                - 0.5 * np.einsum('d,kde,e->k', offset, self.precision, offset))
        return TransferMixture(precision, natural, bias, self.expert.copy())


def compose_values(front, back):
    return np.concatenate([front[..., :3] + front[..., 3:4] * back[..., :3],
                           front[..., 3:4] * back[..., 3:4]], axis=-1)


def random_mixture(rng, k, d=4):
    a = rng.normal(size=(k, d, d)) * 0.4
    precision = a @ np.swapaxes(a, 1, 2) + 0.25 * np.eye(d)[None]
    mean = rng.normal(size=(k, d))
    natural = np.einsum('kde,ke->kd', precision, mean)
    bias = rng.normal(size=k) - 0.5 * np.einsum('kd,kde,ke->k', mean, precision, mean)
    t = rng.uniform(0.02, 0.98, size=(k, 1))
    c = rng.uniform(0, 1, size=(k, 3))
    return TransferMixture(precision, natural, bias, np.concatenate([(1-t)*c, t], axis=1))


def algebra_checks():
    rng = np.random.default_rng(20261003)
    a, b, c = (random_mixture(rng, k) for k in (5, 6, 4))
    q = rng.uniform(-2, 2, size=(2048, 4))
    ab = a.compose(b)
    explicit = compose_values(a.evaluate(q), b.evaluate(q))
    closure_error = float(np.max(np.abs(ab.evaluate(q) - explicit)))
    left, right = a.compose(b).compose(c), a.compose(b.compose(c))
    assoc_error = float(np.max(np.abs(left.evaluate(q) - right.evaluate(q))))
    value, jac = ab.evaluate(q[:32], jacobian=True)
    fd = np.empty_like(jac)
    h = 1e-5
    for d in range(4):
        dq = np.zeros(4); dq[d] = h
        fd[:, :, d] = (ab.evaluate(q[:32]+dq)-ab.evaluate(q[:32]-dq))/(2*h)
    jac_error = float(np.max(np.abs(jac-fd)))
    matrix, offset = rng.normal(size=(4, 2)), rng.normal(size=4)
    u = rng.normal(size=(256, 2))
    sliced = a.restrict_affine(matrix, offset)
    slice_error = float(np.max(np.abs(a.evaluate(u@matrix.T+offset)-sliced.evaluate(u))))
    all_values = left.evaluate(q)
    physical_violation = float(max(0, -all_values.min(),
                                    np.max(all_values[:, 3]-1),
                                    np.max(all_values[:, :3]+all_values[:, 3:4]-1)))
    # Front red, middle green, back blue, each opacity 0.5.
    red, green, blue = np.array([.5,0,0,.5]), np.array([0,.5,0,.5]), np.array([0,0,.5,.5])
    valid = compose_values(compose_values(red, green), blue)
    interleaved_wrong = compose_values(compose_values(red, blue), green)
    # Small values can hide large derivative errors even with physical RGB/T.
    x = np.linspace(-1, 1, 20001)
    amp, frequency = 1e-4, 1e4
    tiny_value_error = amp*np.sin(frequency*x)
    large_derivative_error = amp*frequency*np.cos(frequency*x)
    checks = dict(
        exact_product_components=30, associativity_components=120,
        product_closure_max_abs=closure_error,
        associativity_max_abs=assoc_error,
        affine_restriction_max_abs=slice_error,
        input_jacobian_vs_central_difference_max_abs=jac_error,
        physical_constraint_max_violation=physical_violation,
        interleaving_correct_rgba=valid.tolist(),
        interleaving_wrong_rgba=interleaved_wrong.tolist(),
        interleaving_max_abs_error=float(np.max(np.abs(valid-interleaved_wrong))),
        small_value_large_gradient_example={
            'max_value_error':float(np.max(np.abs(tiny_value_error))),
            'max_derivative_error':float(np.max(np.abs(large_derivative_error))),
        },
    )
    assert closure_error < 1e-12 and assoc_error < 1e-12 and slice_error < 1e-12
    assert jac_error < 1e-7 and physical_violation < 1e-12
    return checks


def synthetic_scene(kind):
    """64 Gaussian-opacity sheets in flatland; not a 3DGS dataset."""
    rng = np.random.default_rng(407 if kind == 'smooth' else 409)
    n = 64
    z = np.linspace(0.05, 0.95, n)
    if kind == 'smooth':
        center = rng.uniform(-0.35, 0.35, n)
        sigma = rng.uniform(0.45, 0.75, n)
        opacity = rng.uniform(0.035, 0.07, n)
        color = np.column_stack([0.30+0.18*z, 0.60-0.12*z, 0.40+0.1*z])
    else:
        center = np.linspace(-1.2, 1.2, n)
        rng.shuffle(center)
        sigma = rng.uniform(0.012, 0.025, n)
        opacity = rng.uniform(0.75, 0.96, n)
        color = rng.uniform(0, 1, (n, 3))
    return center, sigma, opacity, color, z


def teacher(q, scene, jacobian=False):
    center, sigma, opacity, color, depth = scene
    # q=(horizontal intercept, normalized ray slope).
    x = q[:, 0:1]+0.5*q[:, 1:2]*depth[None]
    alpha = opacity[None]*np.exp(-0.5*((x-center[None])/sigma[None])**2)
    e = np.zeros((len(q), 3)); t = np.ones((len(q), 1))
    if jacobian:
        de = np.zeros((len(q), 3, 2)); dt = np.zeros((len(q), 1, 2))
    for k in range(len(center)):
        a = alpha[:, k:k+1]
        if jacobian:
            dax = -a*(x[:, k:k+1]-center[k])/sigma[k]**2
            da = np.stack([dax[:, 0], 0.5*depth[k]*dax[:, 0]], axis=-1)[:, None]
            de = de + (dt*a[..., None]+t[..., None]*da)*color[k][None, :, None]
            dt = dt*(1-a[..., None])-t[..., None]*da
        e = e + t*a*color[k]
        t = t*(1-a)
    result = np.concatenate([e, t], axis=1)
    return (result, np.concatenate([de, dt], axis=1)) if jacobian else result


def expert_values(p):
    t, color = expit(p[:, 6:7]), expit(p[:, 7:10])
    return np.concatenate([(1-t)*color, t], axis=1)


def predict_and_vjp(p, q, cotangent=None):
    """Learned full 2D Gaussian gates; manually differentiated reference."""
    dx, dy = q[:, 0:1]-p[None, :, 0], q[:, 1:2]-p[None, :, 1]
    l00, l10, l11 = np.exp(p[:, 2]), p[:, 3], np.exp(p[:, 4])
    z0, z1 = dx*l00+dy*l10, dy*l11
    weights = softmax(p[None, :, 5]-0.5*(z0*z0+z1*z1), axis=1)
    expert = expert_values(p)
    pred = weights@expert
    if cotangent is None:
        return pred
    gl = weights * (cotangent@expert.T-np.sum(cotangent*pred,axis=1,keepdims=True))
    grad = np.zeros_like(p)
    grad[:, 0] = np.sum(gl*z0*l00, axis=0)
    grad[:, 1] = np.sum(gl*(z0*l10+z1*l11), axis=0)
    grad[:, 2] = np.sum(-gl*z0*dx*l00, axis=0)
    grad[:, 3] = np.sum(-gl*z0*dy, axis=0)
    grad[:, 4] = np.sum(-gl*z1*dy*l11, axis=0)
    grad[:, 5] = gl.sum(axis=0)
    gv = weights.T@cotangent
    t, color = expit(p[:, 6:7]), expit(p[:, 7:10])
    grad[:, 6] = ((gv[:, 3:4]-np.sum(gv[:, :3]*color,axis=1,keepdims=True))*t*(1-t))[:, 0]
    grad[:, 7:10] = gv[:, :3]*(1-t)*color*(1-color)
    return pred, grad


def to_mixture(p):
    k = len(p)
    l = np.zeros((k, 2, 2))
    l[:, 0, 0], l[:, 1, 0], l[:, 1, 1] = np.exp(p[:,2]), p[:,3], np.exp(p[:,4])
    precision = l@np.swapaxes(l,1,2)
    natural = np.einsum('kde,ke->kd',precision,p[:, :2])
    bias = p[:,5]-0.5*np.einsum('kd,kde,ke->k',p[:, :2],precision,p[:, :2])
    return TransferMixture(precision,natural,bias,expert_values(p))


def initialize(k, scene, seed):
    rng = np.random.default_rng(seed)
    p = np.zeros((k, 10))
    p[:, :2] = rng.uniform(-1, 1, (k, 2))
    p[:, 2] = p[:, 4] = np.log(np.sqrt(k)/1.4)
    value = teacher(p[:, :2], scene)
    p[:,6] = logit(np.clip(value[:,3],0.005,0.995))
    p[:,7:10] = logit(np.clip(value[:,:3]/np.maximum(1-value[:,3:4],1e-8),0.005,0.995))
    return p


def objective(flat, k, q, y, derivative_data=None, derivative_weight=0.0):
    p = flat.reshape(k,10)
    pred = predict_and_vjp(p,q)
    err = pred-y
    loss = float(np.mean(err**2))
    _, grad = predict_and_vjp(p,q,2*err/err.size)
    if derivative_data is not None and derivative_weight > 0:
        # Finite-difference Jacobian supervision; not a certified derivative bound.
        h, target_d = derivative_data
        for d in range(2):
            dq = np.zeros(2); dq[d]=h
            pp, pm = predict_and_vjp(p,q+dq), predict_and_vjp(p,q-dq)
            ed = (pp-pm)/(2*h)-target_d[:,:,d]
            loss += derivative_weight*float(np.mean(ed**2))/2
            gp = derivative_weight*ed/(ed.size*2*h)
            _, ga = predict_and_vjp(p,q+dq,gp)
            _, gb = predict_and_vjp(p,q-dq,-gp)
            grad += ga+gb
    return loss, grad.ravel()


def gradient_check():
    rng = np.random.default_rng(92)
    scene=synthetic_scene('smooth'); p=initialize(4,scene,94)
    q=rng.uniform(-1,1,(24,2)); y,dy=teacher(q,scene,True)
    f=p.ravel().copy(); args=(4,q,y,(1e-3,dy),0.02)
    _,g=objective(f,*args)
    errors=[]
    for j in range(len(f)):
        h=1e-5; plus=f.copy(); minus=f.copy(); plus[j]+=h; minus[j]-=h
        fd=(objective(plus,*args)[0]-objective(minus,*args)[0])/(2*h)
        errors.append(abs(fd-g[j]))
    maxerr=float(max(errors)); assert maxerr < 1e-7
    return maxerr


def run_fit(kind, k, derivative_weight, iterations, out):
    scene = synthetic_scene(kind)
    rng = np.random.default_rng(42)
    q = rng.uniform(-1,1,(1024,2)); qt = rng.uniform(-1,1,(4096,2))
    y,dy = teacher(q,scene,True); yt,dyt = teacher(qt,scene,True)
    p=initialize(k,scene,100+k)
    bounds_one=[(-3,3),(-3,3),(-3,4),(-15,15),(-3,4),(-12,12),(-12,12)]+[(-12,12)]*3
    result=minimize(objective,p.ravel(),args=(k,q,y,(1e-3,dy),derivative_weight),
                    method='L-BFGS-B',jac=True,bounds=bounds_one*k,
                    options={'maxiter':iterations,'ftol':1e-13,'gtol':1e-7,'maxls':30})
    learned=to_mixture(result.x.reshape(k,10))
    pred,jac=learned.evaluate(qt,True)
    # Unseen rays, all within the training ray box; no novel 3D views are claimed.
    backgrounds=np.array([[0,0,0],[1,1,1],[0.2,0.6,0.9]])
    expected=yt[:,None,:3]+yt[:,None,3:4]*backgrounds[None]
    rendered=pred[:,None,:3]+pred[:,None,3:4]*backgrounds[None]
    mse=float(np.mean((rendered-expected)**2))
    physical=max(0,float(-pred.min()),float(np.max(pred[:,:3]+pred[:,3:4]-1)),float(np.max(pred[:,3]-1)))
    metric={
        'scene':kind,'teacher_gaussian_sheets':64,'mixture_components':k,
        'derivative_loss_weight':derivative_weight,
        'teacher_to_student_component_ratio':64/k,
        'heldout_ray_transfer_mse':float(np.mean((pred-yt)**2)),
        'heldout_ray_render_psnr_db':float(-10*np.log10(max(mse,1e-30))),
        'heldout_ray_render_max_abs':float(np.max(np.abs(rendered-expected))),
        'heldout_ray_jacobian_relative_l2':float(np.linalg.norm(jac-dyt)/np.linalg.norm(dyt)),
        'physical_constraint_max_violation':physical,
        'optimizer_success':bool(result.success),'optimizer_message':str(result.message),
        'optimizer_iterations':int(result.nit),'training_objective':float(result.fun),
        'not_measured':['GPU latency','3D heldout-view quality','training time to target quality'],
    }
    np.savez(out/f'{kind}_r{k}_d{derivative_weight:g}.npz',parameters=result.x.reshape(k,10))
    assert physical < 1e-12
    return metric,learned


def make_plot(records, models, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes=plt.subplots(2,2,figsize=(11.0,7.0),constrained_layout=True)
    colors={'smooth':'#168582','occlusion':'#d26d43'}
    for kind in ['smooth','occlusion']:
        rows=[x for x in records if x['scene']==kind and x['derivative_loss_weight']==0]
        axes[0,0].plot([r['mixture_components'] for r in rows],[r['heldout_ray_render_psnr_db'] for r in rows],'-o',label=kind,color=colors[kind])
        axes[0,1].plot([r['mixture_components'] for r in rows],[r['heldout_ray_jacobian_relative_l2'] for r in rows],'-o',label=kind,color=colors[kind])
    axes[0,0].set(xlabel='Student Gaussian transfer components',ylabel='Synthetic heldout-ray PSNR (dB)',xticks=[4,8,16,32])
    axes[0,1].set(xlabel='Student Gaussian transfer components',ylabel='Relative ray-Jacobian error',xticks=[4,8,16,32])
    axes[0,0].legend(); axes[0,1].legend()
    q=np.column_stack([np.linspace(-1,1,1000),np.full(1000,0.23)])
    for ax,kind in zip(axes[1],['smooth','occlusion']):
        y=teacher(q,synthetic_scene(kind)); pred=models[(kind,8,0.0)].evaluate(q)
        ax.plot(q[:,0],y[:,0],color='#192c3a',label='64-sheet teacher')
        ax.plot(q[:,0],pred[:,0],color=colors[kind],label='8-component student',linestyle='--')
        ax.set(xlabel='Ray intercept (fixed slope = 0.23)',ylabel='Red, black background',title=kind)
        ax.legend(fontsize=9)
    fig.suptitle('CGT feasibility probe: synthetic flatland, CPU only\nComponent reduction is NOT measured GPU speedup',fontsize=14,fontweight='bold')
    fig.savefig(out/'synthetic_probe.png',dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='results'); parser.add_argument('--iterations',type=int,default=500)
    args=parser.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    algebra=algebra_checks(); algebra['parameter_vjp_check_max_abs']=gradient_check()
    print('Algebra and VJP checks:',json.dumps(algebra),flush=True)
    records=[]; models={}
    for kind in ['smooth','occlusion']:
        for k in [4,8,16,32]:
            metric,model=run_fit(kind,k,0.0,args.iterations,out)
            records.append(metric); models[(kind,k,0.0)]=model
            print(json.dumps(metric),flush=True)
    metric,model=run_fit('smooth',8,0.02,args.iterations,out)
    records.append(metric); models[('smooth',8,0.02)]=model
    print(json.dumps(metric),flush=True)
    data={'status':'ALGEBRA_VALIDATED_SYNTHETIC_ONLY',
          'seed_policy':'Fixed seeds in source, one optimization run per condition, no best-of-seed selection',
          'limitations':['2D ray domain, not a 3D scene','Random heldout rays within the same ray domain, not heldout camera paths',
                         'No CUDA code or real GPU timings','Finite optimizer budget, no optimality guarantee',
                         'No general low-rank guarantee','No energy or memory benchmark'],
          'algebra':algebra,'synthetic_fits':records}
    (out/'results.json').write_text(json.dumps(data,indent=2)+'\n')
    make_plot(records,models,out)


if __name__=='__main__':
    main()

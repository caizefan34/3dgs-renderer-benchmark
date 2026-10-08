"""Real-checkpoint, fixed-geometry appearance oracle on tiny training crops.

Independent NumPy EWA renderer; NOT parity-certified against gsplat. This probe
measures a restricted DC correction with all other attributes fixed, not training
speed, reconstruction quality of a trained method, or a leaderboard result.
"""

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.ndimage import correlate
from scipy.special import expit
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
C0 = 0.28209479177387814


def rotations(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


def columns(vertices, names):
    return np.stack([vertices[n] for n in names], -1).astype(np.float64)


def crop_fixture(vertices, camera, images, eval_sh, width, size, crop_xy):
    height = round(camera['height']*width/camera['width'])
    fx, fy = camera['fx']*width/camera['width'], camera['fy']*height/camera['height']
    rotation = np.asarray(camera['rotation'])
    center = np.asarray(camera['position'])
    x0, y0 = round(crop_xy[0]*width)-size//2, round(crop_xy[1]*height)-size//2
    yy, xx = np.mgrid[y0:y0+size, x0:x0+size]
    pixels = np.stack((xx.ravel(), yy.ravel()), -1)
    gathered = []
    for start in range(0, len(vertices), 100000):
        v = vertices[start:start+100000]
        means = columns(v, ['x', 'y', 'z'])
        p = (means-center) @ rotation
        valid = p[:, 2] > 0.2
        ids = np.flatnonzero(valid)
        p, v, means = p[valid], v[valid], means[valid]
        x, y, z = p.T
        # Conventional EWA frustum clamp affects covariance, not projected mean.
        tx = np.clip(x/z, -1.3*width/(2*fx), 1.3*width/(2*fx))
        ty = np.clip(y/z, -1.3*height/(2*fy), 1.3*height/(2*fy))
        zero = np.zeros_like(z)
        j = np.stack((fx/z, zero, -fx*tx/z, zero, fy/z, -fy*ty/z), -1).reshape(-1, 2, 3)
        rq = rotations(columns(v, ['rot_0', 'rot_1', 'rot_2', 'rot_3']))
        scale = np.exp(columns(v, ['scale_0', 'scale_1', 'scale_2']))
        m = (j @ rotation.T @ rq)*scale[:, None, :]
        cov = m @ m.transpose(0, 2, 1)+0.3*np.eye(2)
        opacity = expit(v['opacity'].astype(np.float64))
        xy = np.stack((fx*x/z+width/2, fy*y/z+height/2), -1)
        maxeig = 0.5*(cov[:, 0, 0]+cov[:, 1, 1]+np.sqrt((cov[:, 0, 0]-cov[:, 1, 1])**2+4*cov[:, 0, 1]**2))
        radius = np.sqrt(np.maximum(0, 2*np.log(np.maximum(opacity*255, 1)))*maxeig)
        hit = ((opacity >= 1/255) & (xy[:, 0]+radius >= x0) & (xy[:, 0]-radius <= x0+size-1)
               & (xy[:, 1]+radius >= y0) & (xy[:, 1]-radius <= y0+size-1))
        gathered.append((ids[hit]+start, xy[hit], z[hit], cov[hit], opacity[hit], means[hit]))
    ids, xy, depth, cov, opacity, means = (np.concatenate([part[i] for part in gathered]) for i in range(6))
    order = np.argsort(depth, kind='stable')
    ids, xy, cov, opacity, means = ids[order], xy[order], cov[order], opacity[order], means[order]
    delta = pixels[:, None, :]-xy[None, :, :]
    power = np.einsum('pni,nij,pnj->pn', delta, np.linalg.inv(cov), delta, optimize=True)
    alpha = np.minimum(0.99, opacity[None, :]*np.exp(-0.5*power))
    alpha[alpha < 1/255] = 0
    trans = np.cumprod(np.concatenate((np.ones((len(pixels), 1)), 1-alpha), 1), 1)
    weights = alpha*trans[:, :-1]
    weights[trans[:, 1:] < 1e-4] = 0  # declared independent-renderer termination
    v = vertices[ids]
    dc = columns(v, ['f_dc_0', 'f_dc_1', 'f_dc_2'])[:, :, None]
    rest = columns(v, [f'f_rest_{i}' for i in range(45)]).reshape(-1, 3, 15)
    direction = means-center
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    rgb = eval_sh(3, np.concatenate((dc, rest), -1), direction)+0.5
    baseline = weights @ np.maximum(rgb, 0)
    files = list(images.glob(camera['img_name']+'.*'))
    if len(files) != 1:
        raise ValueError(f"Expected one source image: {camera['img_name']}, got {files}")
    im = Image.open(files[0]).convert('RGB').resize((width, height), Image.Resampling.LANCZOS)
    target = np.asarray(im, dtype=np.float64)[y0:y0+size, x0:x0+size].reshape(-1, 3)/255
    return dict(ids=ids, weights=weights, rgb=rgb, baseline=baseline, target=target,
                image=camera['img_name'], candidates=len(ids), pixels=size*size,
                crop=[x0, y0, size, size], image_sha256=hashlib.sha256(files[0].read_bytes()).hexdigest())


def configure_active(views, active):
    for view in views:
        lookup = {int(g): j for j, g in enumerate(view['ids'])}
        w = np.zeros((view['pixels'], len(active)))
        rgb = np.zeros((len(active), 3))
        for k, g in enumerate(active):
            if int(g) in lookup:
                j = lookup[int(g)]
                w[:, k], rgb[k] = view['weights'][:, j], view['rgb'][j]
        view['active_w'], view['active_rgb'] = w, rgb
        view['fixed'] = view['baseline']-w @ np.maximum(rgb, 0)


def predictions(views, delta):
    return [v['fixed']+v['active_w'] @ np.maximum(v['active_rgb']+C0*delta, 0) for v in views]


def cost(views, delta, damping):
    pred = predictions(views, delta)
    return 0.5*sum(np.sum((p-v['target'])**2) for p, v in zip(pred, views))/sum(v['pixels'] for v in views)+0.5*damping*np.sum(delta**2)


def metrics(views, delta, size):
    k = np.exp(-0.5*(np.arange(-5, 6)/1.5)**2)
    k /= k.sum()
    kernel = k[:, None, None]*k[None, :, None]
    mse, l1, ssim = [], [], []
    for p, v in zip(predictions(views, delta), views):
        p, gt = np.clip(p.reshape(size, size, 3), 0, 1), v['target'].reshape(size, size, 3)
        mu, mg = correlate(p, kernel, mode='constant'), correlate(gt, kernel, mode='constant')
        vp = correlate(p*p, kernel, mode='constant')-mu*mu
        vg = correlate(gt*gt, kernel, mode='constant')-mg*mg
        cross = correlate(p*gt, kernel, mode='constant')-mu*mg
        structure = ((2*mu*mg+0.01**2)*(2*cross+0.03**2))/((mu*mu+mg*mg+0.01**2)*(vp+vg+0.03**2))
        ssim.append(float(structure[5:-5, 5:-5].mean()))
        mse.append(float(np.mean((p-gt)**2)))
        l1.append(float(np.mean(np.abs(p-gt))))
    return {'crop_psnr_mean': float(np.mean(-10*np.log10(np.maximum(mse, 1e-20)))),
            'crop_l1_mean': float(np.mean(l1)), 'crop_ssim_valid_mean': float(np.mean(ssim)),
            'per_view_psnr': [-10*float(np.log10(max(x, 1e-20))) for x in mse]}


def run(args):
    checkpoint = ROOT/f'data/official/mipnerf360/{args.scene}/point_cloud.ply'
    camera_path = checkpoint.with_name('cameras.json')
    images = ROOT/f'data/datasets/mipnerf360/{args.scene}/images'
    sh_path = ROOT/'public_baselines/fastgs/utils/sh_utils.py'
    spec = importlib.util.spec_from_file_location('reference_sh', sh_path)
    sh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sh)
    vertices = PlyData.read(str(checkpoint), mmap='r')['vertex'].data
    cameras = sorted(json.loads(camera_path.read_text()), key=lambda c: c['img_name'])
    # Conventional LLFF every-eighth split; record inference rather than certify it.
    train = [c for i, c in enumerate(cameras) if i % 8 != 0]
    selected = [train[i] for i in (0, 2, 4, 6)]
    started = time.perf_counter()
    views = []
    for camera in selected:
        v = crop_fixture(vertices, camera, images, sh.eval_sh, args.width, args.size, (args.crop_x, args.crop_y))
        views.append(v)
        print(f"{args.scene}/{v['image']}: {v['candidates']} crop candidates", flush=True)
    fit, validation = views[:3], views[3:]
    score = np.zeros(len(vertices))
    for v in fit:
        score[v['ids']] += v['weights'].sum(0)
    active = np.argsort(score)[-args.active:]
    configure_active(views, active)
    build_s = time.perf_counter()-started
    zero = np.zeros((len(active), 3))
    solve_start = time.perf_counter()
    rows = sum(v['pixels'] for v in fit)
    target = np.concatenate([v['target']-v['baseline'] for v in fit])
    candidate = np.zeros_like(zero)
    conditions = []
    offdiag = []
    for channel in range(3):
        a = np.concatenate([C0*v['active_w']*(v['active_rgb'][None, :, channel] > 0) for v in fit])
        h = a.T @ a/rows+args.damping*np.eye(len(active))
        candidate[:, channel] = np.linalg.solve(h, a.T @ target[:, channel]/rows)
        conditions.append(float(np.linalg.cond(h)))
        offdiag.append(float(np.linalg.norm(h-np.diag(h.diagonal()))/np.linalg.norm(h)))
    accepted, step = zero, 0.0
    initial = cost(fit, zero, args.damping)
    for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
        if cost(fit, scale*candidate, args.damping) < initial:
            accepted, step = scale*candidate, scale
            break
    solve_s = time.perf_counter()-solve_start
    adam_start = time.perf_counter()
    delta, m, v2 = zero.copy(), zero.copy(), zero.copy()
    for iteration in range(1, 101):
        grad = args.damping*delta
        for view, p in zip(fit, predictions(fit, delta)):
            mask = view['active_rgb']+C0*delta > 0
            grad = grad+C0*(view['active_w'].T @ (p-view['target']))*mask/rows
        m, v2 = 0.9*m+0.1*grad, 0.999*v2+0.001*grad*grad
        delta -= 0.05*(m/(1-0.9**iteration))/(np.sqrt(v2/(1-0.999**iteration))+1e-8)
    adam_s = time.perf_counter()-adam_start
    with checkpoint.open('rb') as checkpoint_file:
        checkpoint_hash = hashlib.file_digest(checkpoint_file, 'sha256').hexdigest()
    output = {
        'scope': 'DIAGNOSTIC_ONLY: pretrained fixed geometry, restricted DC correction, independent CPU crop renderer; not gsplat parity or E2E training',
        'scene': args.scene, 'checkpoint_vertices': len(vertices), 'checkpoint_bytes': checkpoint.stat().st_size,
        'checkpoint_sha256': checkpoint_hash,
        'probe_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'numpy_version': np.__version__,
        'camera_sha256': hashlib.sha256(camera_path.read_bytes()).hexdigest(),
        'sh_source_sha256': hashlib.sha256(sh_path.read_bytes()).hexdigest(),
        'split': 'sorted camera img_name, index%8!=0; conventional split inferred, not independently certified; validation is from same training subset',
        'parameters': vars(args), 'active_gaussian_ids': active.tolist(),
        'views': [{k: v[k] for k in ('image', 'candidates', 'pixels', 'crop', 'image_sha256')} for v in views],
        'timing_cpu_seconds': {'fixture_and_operator_build': build_s, 'joint_solve_and_acceptance': solve_s, 'cached_adam100': adam_s},
        'timing_warning': 'Both optimizers use the same cached crop operators. Excludes full-frame CUDA rasterization; Adam lr=0.05 fixed, not a tuned baseline.',
        'regularized_fit_objective': {'initial': initial, 'joint': cost(fit, accepted, args.damping), 'adam100': cost(fit, delta, args.damping)},
        'accepted_step': step, 'hessian_condition_rgb': conditions, 'hessian_offdiagonal_fraction_rgb': offdiag,
        'metrics': {name: {'fit': metrics(fit, d, args.size), 'validation': metrics(validation, d, args.size)}
                    for name, d in [('initial', zero), ('joint', accepted), ('adam100', delta)]},
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({k: output[k] for k in ('regularized_fit_objective', 'timing_cpu_seconds', 'accepted_step', 'metrics')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='room', choices=['room', 'bicycle', 'garden'])
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--size', type=int, default=26)
    parser.add_argument('--active', type=int, default=32)
    parser.add_argument('--damping', type=float, default=1e-4)
    parser.add_argument('--crop-x', type=float, default=0.5)
    parser.add_argument('--crop-y', type=float, default=0.5)
    parser.add_argument('--output', default=str(Path(__file__).with_name('room_crop_evidence.json')))
    with threadpool_limits(limits=1):
        run(parser.parse_args())

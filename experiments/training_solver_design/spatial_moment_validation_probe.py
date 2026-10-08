"""Real-state gradient-variance oracle. Fine rendering remains dense on purpose.

No training, sparse-kernel timing, or speedup claim. Original parameters/history
are restored, never advanced; standard Adam updates are evaluated as diagnostics.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from multilevel_loss import block_means, fused_loss_map
from real_topology_probe import sha
from spatial_moment_predictor import spatial_nodes, allocate_nodes
from sort_reuse_renderer import prepare, intersections


@torch.no_grad()
def adam_update(gradient, group, state):
    b1, b2 = group['betas']
    step = float(state['step'])+1
    m = (b1*state['exp_avg']+(1-b1)*gradient)/(1-b1**step)
    v = (b2*state['exp_avg_sq']+(1-b2)*gradient.square())/(1-b2**step)
    return -group['lr']*m/(v.sqrt()+group['eps'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--gpu', type=int, default=6)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.set_num_threads(4)
    sys.path.insert(0, '/root/epic05_p11')
    import p11_common as C
    gpu = C.assert_clean_gpu(args.gpu)
    uuid = gpu['gpu_info'].split(',')[0].strip()
    torch.cuda.init()
    torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    driver_rows = [r for r in C.gpu_compute_apps() if uuid in r]
    assert len(driver_rows) == 1
    driver_pid = driver_rows[0].split(',')[0].strip()
    gpu.update(driver_pid=driver_pid, namespace_pid=os.getpid())

    def exclusive():
        rows = [r for r in C.gpu_compute_apps() if uuid in r]
        assert {r.split(',')[0].strip() for r in rows} == {driver_pid}, rows
        return rows

    started = time.perf_counter()
    identity = C.bootstrap_gmain()
    C.import_recipe()
    from gaussian_model import GaussianModel
    import fused_ssim
    root = Path(__file__).parent
    state_path = Path(C.STATEGEN_DIR)/args.scene/'state_2000.pt'
    previous = json.loads((root/f'{args.scene}_2000_topology.json').read_text())
    assert sha(state_path) == previous['checkpoint_sha256']
    saved = torch.load(state_path, map_location='cpu', weights_only=False)
    model = GaussianModel(max_sh_degree=3)
    model.restore(saved, C.TRAIN_CFG)
    model.optimizer.load_state_dict(copy.deepcopy(saved['optimizer_state_dict']))
    model.update_learning_rate(2001)
    groups = model.optimizer.param_groups
    params = [g['params'][0] for g in groups]
    states = [model.optimizer.state[p] for p in params]
    dataset = C.load_dataset(args.scene)
    result = {'scope': 'Dense-render diagnostic of spatial-moment oracle predictor; fresh same-view gradient is unattainable oracle, not algorithm; no speed/training result',
              'scene': args.scene, 'N': saved['num_points'], 'source_sha256': sha(__file__),
              'helper_sha256': sha(root/'multilevel_loss.py'), 'predictor_helper_sha256': sha(root/'spatial_moment_predictor.py'), 'projection_helper_sha256': sha(root/'sort_reuse_renderer.py'), 'checkpoint_sha256': sha(state_path),
              'common_sha256': sha(C.__file__), 'renderer_identity': identity,
              'fused_ssim_python_sha256': sha(fused_ssim.__file__), 'gpu': gpu,
              'protocol': {'views': previous['A'], 'fine_shape': [1080, 1920], 'block_grid': [9, 15],
                           'predictors': ['spatial_same_view_fresh_oracle'], 'partition_seeds': [701, 702],
                           'sampling': 'uniform choice of one of four random groups; marginal inclusion 1/4',
                           'beta': .5, 'beta_selection': 'chosen from garden development sums before this room validation; not preregistered before garden', 'allocation': 'signed degree-2 Gaussian-Hermite moment reconstruction at 9 nodes; positions clipped to image blocks; invalid rows zero', 'eps2d': C.EPS2D,
                           'oracle_beta': 'estimated from all audit draws; unattainable selection bound, not a deployed rule',
                           'Adam_metric': 'hypothetical standard Adam step difference with saved moments; not a training trajectory',
                           'B_or_test_views_used': False, 'fine_render_is_dense': True}, 'records': []}

    def save():
        result['wall_seconds'] = time.perf_counter()-started
        Path(args.out).write_text(json.dumps(result, indent=2)+'\n')

    for camera_id in previous['A']:
        exclusive()
        cam, gt = dataset.get_item(camera_id)
        assert (cam.image_height, cam.image_width) == (1080, 1920)
        from gsplat.cuda._wrapper import rasterize_to_pixels
        data=prepare(model,cam,C.EPS2D)
        lists=intersections(data,cam.image_width,cam.image_height)
        latents=tuple(data[key] for key in ('xy','conics','opacity','colors'))
        image,_=rasterize_to_pixels(latents[0],latents[1],latents[3],latents[2],cam.image_width,cam.image_height,
                                   16,lists[3],lists[2],packed=False,absgrad=False)
        fine_image=image[0].clamp(0,1)
        _,native_image,_,_=C.render_with_meta(model,cam,model.active_sh_degree,absgrad=False)
        assert torch.equal(fine_image,native_image)

        fine_map = fused_loss_map(fine_image, gt)
        fine_blocks = block_means(fine_map, 9, 15)
        original_loss = C.fused_total_loss(native_image, gt)
        scalar_error = abs(float((fine_blocks.mean()-original_loss).detach()))
        assert scalar_error < 5e-7, scalar_error
        fine_grad = [g.detach() for g in torch.autograd.grad(fine_blocks.mean(), params, retain_graph=True)]
        original_grad = torch.autograd.grad(original_loss, params, retain_graph=True)
        parity = {group['name']: {'relative_l2': float((a-b).norm()/b.norm().clamp_min(1e-20)),
                                  'max_abs': float((a-b).abs().max())}
                  for group, a, b in zip(groups, fine_grad, original_grad)}
        for v in parity.values():
            assert v['relative_l2']<.003 or v['max_abs']<1e-6,v
        del original_grad, original_loss,native_image
        reference_updates = [adam_update(g, group, state) for g, group, state in zip(fine_grad, groups, states)]

        latent_grad=torch.autograd.grad(fine_blocks.mean(),latents,retain_graph=True)
        visible=(data['radii'][0]>0).all(-1)
        nodes,valid=spatial_nodes(data['conics'][0],data['opacity'][0],latent_grad[0][0],latent_grad[1][0],latent_grad[2][0],visible)
        reconstruction={}
        for node_index,target in ((1,latent_grad[0][0]),(2,latent_grad[1][0]),(3,latent_grad[2][0])):
            actual=sum(node[node_index] for node in nodes)
            expected=target*valid.reshape((-1,)+(1,)*(target.ndim-1))
            err={'relative_l2':float((actual-expected).norm()/expected.norm().clamp_min(1e-20)),
                 'max_abs':float((actual-expected).abs().max())}
            reconstruction[node_index]=err
            assert err['relative_l2']<.003 or err['max_abs']<1e-6,err
        for predictor_name in ('spatial_same_view_fresh_oracle',):
            exclusive()
            proxy_loss=sum((latent*grad.detach()*valid.reshape((1,-1)+(1,)*(latent.ndim-2))).sum() for latent,grad in zip(latents,latent_grad))
            proxy_grad=list(torch.autograd.grad(proxy_loss,params,retain_graph=True))
            accum = [{key: 0. for key in ('base_sq', 'cv_sq', 'proxy_sq', 'cross', 'base_adam_sq', 'cv_adam_sq')}
                     for _ in params]
            mean_base = [torch.zeros_like(g) for g in fine_grad]
            mean_cv = [torch.zeros_like(g) for g in fine_grad]
            draws = []
            for seed in (701, 702):
                generator = torch.Generator(device='cpu').manual_seed(seed)
                labels = (torch.randperm(135, generator=generator).reshape(9, 15) % 4).cuda()
                predicted=allocate_nodes(data['xy'][0],nodes,latent_grad[3][0],valid,labels,cam.image_height,cam.image_width)
                for label in range(4):
                    weight = (labels == label).float()*4
                    gs = torch.autograd.grad((weight*fine_blocks).mean(), params, retain_graph=True)
                    hs = torch.autograd.grad(sum((latent*prediction[None]*4).sum() for latent,prediction in zip(latents,predicted[label])), params, retain_graph=True)
                    draw = {'seed': seed, 'group': label, 'num_blocks': int((labels == label).sum()), 'groups': {}}
                    for i, (g, h, gf, hf, group, state, uref) in enumerate(zip(gs, hs, fine_grad, proxy_grad, groups, states, reference_updates)):
                        with torch.no_grad():
                            error = g-gf
                            proxy_error = h-hf
                            cv_error = error-.5*proxy_error
                            ubase = adam_update(g, group, state)
                            ucv = adam_update(g-.5*proxy_error, group, state)
                            values = {'base_sq': float(error.square().sum()), 'cv_sq': float(cv_error.square().sum()),
                                      'proxy_sq': float(proxy_error.square().sum()), 'cross': float((error*proxy_error).sum()),
                                      'base_adam_sq': float((ubase-uref).square().sum()),
                                      'cv_adam_sq': float((ucv-uref).square().sum())}
                            for key, value in values.items():
                                accum[i][key] += value
                            mean_base[i].add_(error/8)
                            mean_cv[i].add_(cv_error/8)
                            draw['groups'][group['name']] = values
                    del gs, hs, g, h, error, proxy_error, cv_error, ubase, ucv
                    draws.append(draw)
            summary = {}
            for group, gf, values, mb, mc in zip(groups, fine_grad, accum, mean_base, mean_cv):
                beta = values['cross']/max(values['proxy_sq'], 1e-30)
                summary[group['name']] = {
                    'beta_half_variance_ratio': values['cv_sq']/max(values['base_sq'], 1e-30),
                    'oracle_beta': beta,
                    'oracle_variance_ratio': 1-values['cross']**2/max(values['proxy_sq']*values['base_sq'], 1e-30),
                    'beta_half_adam_error_ratio': values['cv_adam_sq']/max(values['base_adam_sq'], 1e-30),
                    'base_mean_error_relative': float(mb.norm()/gf.norm().clamp_min(1e-20)),
                    'cv_mean_error_relative': float(mc.norm()/gf.norm().clamp_min(1e-20)),
                    'gradient_norm': float(gf.norm()), 'sums': values}
            result['records'].append({'camera_id': camera_id, 'predictor': predictor_name, 'moment_reconstruction': reconstruction, 'invalid_visible_rows': int((visible & ~valid).sum()),
                                      'fine_loss': float(fine_blocks.mean().detach()), 'original_loss_error': scalar_error,
                                      'loss_gradient_parity': parity, 'summary': summary, 'draws': draws})
            save()
            print('DONE', args.scene, camera_id, predictor_name,
                  {name: (round(v['beta_half_variance_ratio'], 3), round(v['oracle_beta'], 3), round(v['oracle_variance_ratio'], 3))
                   for name, v in summary.items()}, flush=True)
            del proxy_grad, mean_base, mean_cv, hf, gf, mb, mc
        del fine_image, fine_map, fine_blocks, fine_grad, reference_updates, nodes, predicted, proxy_loss, data, latents, latent_grad,image
    result['gpu_at_end'] = exclusive()
    result['completed'] = True
    save()
    print('COMPLETE', args.scene, args.out, flush=True)


if __name__ == '__main__':
    main()

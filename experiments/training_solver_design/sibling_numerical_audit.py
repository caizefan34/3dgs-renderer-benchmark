"""Numerical replay audit of sibling coordinates; no optimizer experiment."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from newborn_updates import step_with_newborn_policy
from sibling_coordinates import SiblingView, encode_children, decode_raw
from real_topology_probe import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='room')
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--steps', type=int, default=256)
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
    rows = [x for x in C.gpu_compute_apps() if uuid in x]
    assert len(rows) == 1, rows
    driver_pid = rows[0].split(',')[0].strip()
    gpu.update(namespace_pid=os.getpid(), driver_pid=driver_pid)

    def exclusive():
        rows = [x for x in C.gpu_compute_apps() if uuid in x]
        assert {x.split(',')[0].strip() for x in rows} == {driver_pid}, rows
        return rows

    started = time.perf_counter()
    identity = C.bootstrap_gmain()
    C.import_recipe()
    from gaussian_model import GaussianModel
    root = Path(__file__).parent
    candidate_path = root / f'{args.scene}_2000_topology.json'
    previous = json.loads(candidate_path.read_text())
    state_path = Path(C.STATEGEN_DIR) / args.scene / 'state_2000.pt'
    assert sha(state_path) == previous['checkpoint_sha256']
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    n = state['num_points']
    extent = state['spatial_lr_scale']
    parent_ids = sorted({i for a in previous['candidates'] for i in a['ids']})
    assert len(parent_ids) == 256, 'Do not silently change birth budget'
    parent_ids = torch.tensor(parent_ids, device='cuda')
    nparents = len(parent_ids)
    first_newborn = n - nparents
    dataset = C.load_dataset(args.scene)
    a_ids = previous['A']
    b_ids = list(dict.fromkeys(int(i) for i in np.linspace(0, len(dataset)-1, 20)))
    b_ids = [i for i in b_ids if i not in a_ids][:16]
    assert len(b_ids) == 16 and not set(a_ids) & set(b_ids)
    views = {'A': [dataset.get_item(i) for i in a_ids], 'B': [dataset.get_item(i) for i in b_ids]}
    seq_path = state_path.with_name('camera_sequence.npy')
    sequence = np.load(seq_path)[2000:].tolist()
    training_ids = [i for i in sequence if i not in b_ids][:args.steps]
    assert len(training_ids) == args.steps and args.steps >= 128
    horizons = sorted(set([0, 1, 8, 32, 128, args.steps]))
    group_steps = {g['name']: float(state['optimizer_state_dict']['state'][g['params'][0]]['step'])
                   for g in state['optimizer_state_dict']['param_groups']}
    assert len(set(group_steps.values())) == 1

    def restore():
        model = GaussianModel(max_sh_degree=3)
        model.restore(state, C.TRAIN_CFG)
        model.optimizer.load_state_dict(copy.deepcopy(state['optimizer_state_dict']))
        model.xyz_gradient_accum = state['xyz_gradient_accum'].cuda().clone()
        model.denom = state['denom'].cuda().clone()
        for group, saved_group in zip(model.optimizer.param_groups, state['optimizer_state_dict']['param_groups']):
            actual = model.optimizer.state[group['params'][0]]
            expected = state['optimizer_state_dict']['state'][saved_group['params'][0]]
            assert float(actual['step']) == float(expected['step'])
            for key in ('exp_avg', 'exp_avg_sq'):
                assert torch.equal(actual[key].reshape(-1)[:4096].cpu(), expected[key].reshape(-1)[:4096])
        return model

    @torch.no_grad()
    def evaluate(model):
        all_metrics = {}
        for name, group_views in views.items():
            per_view = []
            for cam, gt in group_views:
                _, image, _, _ = C.render_with_meta(model, cam, model.active_sh_degree, absgrad=False)
                loss = float(C.fused_total_loss(image, gt))
                mse = float((image-gt).square().mean())
                per_view.append({'loss': loss, 'mse': mse, 'psnr': -10 * np.log10(mse)})
            all_metrics[name] = {'loss': float(np.mean([v['loss'] for v in per_view])),
                                 'mean_view_psnr': float(np.mean([v['psnr'] for v in per_view])),
                                 'per_view': per_view}
        return all_metrics

    @torch.no_grad()
    def split(model, seed, policy):
        old_states = {g['name']: {k: v.clone() if torch.is_tensor(v) else v
                                  for k, v in model.optimizer.state[g['params'][0]].items()}
                      for g in model.optimizer.param_groups}
        mask = torch.ones(n, device='cuda', dtype=torch.bool)
        mask[parent_ids] = False
        grad = torch.zeros(n, device='cuda')
        grad[parent_ids] = 1
        torch.manual_seed(seed)
        count = model.densify_and_split(grad, 0.5, extent, N=2)
        assert count == nparents and len(model._xyz) == n+nparents
        encoded = policy.startswith('pair_')
        h = hashlib.sha256()
        for group in model.optimizer.param_groups:
            raw = group['params'][0]
            coordinates = encode_children(raw, count)
            raw[-2*count:] = coordinates
            physical = decode_raw(raw, count)
            h.update(physical[first_newborn:].cpu().numpy().tobytes())
            if not encoded:
                raw.copy_(physical)
            current = model.optimizer.state[raw]
            old = old_states[group['name']]
            assert float(current['step']) == float(old['step'])
            for key in ('exp_avg', 'exp_avg_sq'):
                assert torch.equal(current[key][:first_newborn], old[key][mask])
                assert torch.count_nonzero(current[key][first_newborn:]) == 0
                if policy in ('inherit_02', 'inherit_05'):
                    factor = .2 if policy == 'inherit_02' else .5
                    factor = factor if key == 'exp_avg' else factor**2
                    inherited = old[key][parent_ids].repeat((2,) + (1,) * (old[key].ndim-1))
                    current[key][first_newborn:] = factor * inherited
                elif policy in ('pair_shared', 'pair_full'):
                    current[key][first_newborn:first_newborn+count] = old[key][parent_ids]
        return h.hexdigest(), SiblingView(model, count) if encoded else model

    def numerical_audit():
        physical, encoded = restore(), restore()
        hp, vp = split(physical, 20260926, 'native')
        he, ve = split(encoded, 20260926, 'pair_zero')
        assert hp == he
        for name in ('get_xyz', 'get_scaling', 'get_rotation', 'get_opacity', 'get_features'):
            assert torch.equal(getattr(vp, name), getattr(ve, name)), name
        cam, gt = views['A'][0]
        samples = {'native': [], 'encoded': []}
        ref = None
        for repeat in range(4):
            for name, model, view in (('native', physical, vp), ('encoded', encoded, ve)):
                model.optimizer.zero_grad(set_to_none=True)
                _, image, _, _ = C.render_with_meta(view, cam, view.active_sh_degree, absgrad=False)
                if ref is None:
                    ref = image.detach().clone()
                assert torch.equal(image.detach(), ref)
                C.fused_total_loss(image, gt).backward()
                gradients = {}
                for group in model.optimizer.param_groups:
                    g = group['params'][0].grad.detach()
                    if name == 'native':
                        g = torch.cat((g[:-2*nparents], g[-2*nparents:-nparents]+g[-nparents:],
                                       g[-2*nparents:-nparents]-g[-nparents:]))
                    gradients[group['name']] = g.cpu().clone()
                samples[name].append(gradients)
        reports = {}
        for group in samples['native'][0]:
            reports[group] = {}
            for part, selection in (('survivors', slice(None, -2*nparents)), ('children', slice(-2*nparents, None))):
                a = [s[group][selection] for s in samples['native']]
                b = [s[group][selection] for s in samples['encoded']]
                def discrepancy(pairs):
                    return {'max_abs': max(float((x-y).abs().max()) for x,y in pairs),
                            'max_l2': max(float((x-y).norm()) for x,y in pairs)}
                native = discrepancy([(a[i], a[j]) for i in range(4) for j in range(i)])
                changed = discrepancy([(b[i], b[j]) for i in range(4) for j in range(i)])
                cross = discrepancy([(x,y) for x in a for y in b])
                mean_error = discrepancy([(torch.stack(a).mean(0), torch.stack(b).mean(0))])
                reports[group][part] = {'native_repeat': native, 'encoded_repeat': changed,
                                        'cross_basis': cross, 'mean_error': mean_error,
                                        'native_norm': float(a[0].norm())}
        return {'scene': args.scene, 'source_sha256': sha(__file__), 'original_probe_sha256': sha(root/'sibling_probe.py'),
                'coordinate_helper_sha256': sha(root/'sibling_coordinates.py'), 'exact_initial_images': True,
                'checkpoint_sha256': sha(state_path), 'gpu': gpu, 'reports': reports,
                'gpu_at_end': exclusive()}

    result = numerical_audit()
    Path(args.out).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

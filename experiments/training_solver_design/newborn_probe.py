"""Matched real-state controls for newborn Adam state, not a new method claim.

Fixed union of previous candidate groups; four state/update policies, three
matched split draws, native repeat per draw, two no-edit controls. No ranking
or split-initialization change. BOGausS-inspired inheritance is NOT BOGausS.
"""

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
    old_step = next(iter(group_steps.values()))
    beta1, beta2 = state['optimizer_state_dict']['param_groups'][0]['betas']
    first_step_ratio = (1-beta1)/(1-beta1**(old_step+1)) * ((1-beta2**(old_step+1))/(1-beta2))**0.5
    constant_scale = 1 / first_step_ratio

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
        for group in model.optimizer.param_groups:
            current = model.optimizer.state[group['params'][0]]
            old = old_states[group['name']]
            assert float(current['step']) == float(old['step'])
            for key in ('exp_avg', 'exp_avg_sq'):
                assert torch.equal(current[key][:first_newborn], old[key][mask])
                assert torch.count_nonzero(current[key][first_newborn:]) == 0
                if policy == 'inherit_02':
                    inherited = old[key][parent_ids].repeat((2,) + (1,) * (old[key].ndim-1))
                    factor = 0.2 if key == 'exp_avg' else 0.2 ** 2
                    current[key][first_newborn:] = inherited * factor
                    assert torch.equal(current[key][first_newborn:], inherited * factor)
        h = hashlib.sha256()
        for group in model.optimizer.param_groups:
            h.update(group['params'][0][first_newborn:].cpu().numpy().tobytes())
        return h.hexdigest()

    seeds = [20260926, 20260927, 20260928]
    arms = [('no_edit', None, 'native')]
    for seed in seeds:
        for policy in ('native', 'fresh_age', 'inherit_02', 'lr_scale'):
            arms.append((f'{policy}_{seed}', seed, policy))
        arms.append((f'native_repeat_{seed}', seed, 'native'))
    arms.append(('no_edit_repeat', None, 'native'))
    result = {'scope': 'Real-state newborn optimizer mechanism controls; no novelty or complete-training claim',
              'scene': args.scene, 'N': n, 'nparents': nparents, 'checkpoint_sha256': sha(state_path),
              'candidate_source_sha256': sha(candidate_path), 'parent_ids': parent_ids.cpu().tolist(),
              'source_sha256': sha(__file__), 'update_helper_sha256': sha(root/'newborn_updates.py'),
              'renderer_identity': identity, 'common_sha256': sha(C.__file__),
              'gaussian_model_sha256': sha(Path(C.REFERENCE_V1_DIR)/'gaussian_model.py'),
              'gpu': gpu, 'A_ids': a_ids, 'B_ids': b_ids, 'training_camera_sequence': training_ids,
              'old_group_steps': group_steps, 'constant_child_lr_scale': constant_scale,
              'protocol': {'split': 'native initialization; fixed union of all previous eight candidate groups',
                           'policies': ['native', 'fresh_age', 'inherit_02', 'lr_scale'],
                           'seeds': seeds, 'horizons': horizons,
                           'inheritance': '0.2*m, 0.04*v, same group clock; BOGausS-inspired but not its optimizer',
                           'B_seen_before': 'B is excluded only from this continuation; not a formal unseen-view test',
                           'loss': '0.8 L1 + 0.2 fused D-SSIM, same padding, 1920x1080',
                           'frozen_later_topology': True, 'fixed_active_SH_degree': state['active_sh_degree'],
                           'first_step_only_scale_match': 'constant lr_scale matches fresh_age first step ignoring epsilon; it does not track later age'},
              'data_sources': {'camera_json_sha256': sha(dataset.official_dir/'cameras.json'),
                               'camera_sequence_sha256': sha(seq_path),
                               'images': {str(i): sha(dataset._valid_image_paths[i]) for i in sorted(set(a_ids+b_ids+training_ids))}},
              'arms': []}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        result['total_seconds_including_setup'] = time.perf_counter()-started
        temp = out.with_suffix('.tmp')
        temp.write_text(json.dumps(result, indent=2)+'\n')
        temp.replace(out)

    child_hashes = {}
    for name, seed, policy in arms:
        exclusive()
        begin = time.perf_counter()
        model = restore()
        initial_hash = None
        if seed is not None:
            initial_hash = split(model, seed, policy)
            if seed in child_hashes:
                assert child_hashes[seed] == initial_hash, 'Policy changed split initialization'
            child_hashes[seed] = initial_hash
        torch.manual_seed(42)
        history = {'0': evaluate(model)}
        step_time, eval_time = 0.0, 0.0
        for step, camera_id in enumerate(training_ids, 1):
            step_begin = time.perf_counter()
            model.update_learning_rate(2000+step)
            cam, gt = dataset.get_item(camera_id)
            model.optimizer.zero_grad(set_to_none=True)
            _, image, _, _ = C.render_with_meta(model, cam, model.active_sh_degree, absgrad=False)
            loss = C.fused_total_loss(image, gt)
            loss.backward()
            step_with_newborn_policy(model.optimizer, first_newborn if seed is not None else None,
                                     step, policy, constant_scale)
            model.optimizer.zero_grad(set_to_none=True)
            del image, loss
            torch.cuda.synchronize()
            step_time += time.perf_counter()-step_begin
            if step in horizons:
                evaluate_begin = time.perf_counter()
                history[str(step)] = evaluate(model)
                eval_time += time.perf_counter()-evaluate_begin
        result['arms'].append({'name': name, 'seed': seed, 'policy': policy,
                               'initial_children_sha256': initial_hash, 'history': history,
                               'train_seconds_with_sync': step_time, 'evaluation_seconds_after_initial': eval_time,
                               'arm_seconds': time.perf_counter()-begin})
        save()
        print(f'DONE {args.scene} {name} B@{args.steps}={history[str(args.steps)]["B"]["loss"]:.9f} train={step_time:.2f}s arm={time.perf_counter()-begin:.2f}s', flush=True)
        del model
    result['gpu_at_end'] = exclusive()
    result['completed'] = True
    save()
    print(f'COMPLETE {args.scene} {len(arms)} arms: {out}', flush=True)


if __name__ == '__main__':
    main()

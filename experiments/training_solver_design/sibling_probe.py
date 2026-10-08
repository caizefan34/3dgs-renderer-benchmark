"""Matched single-birth sibling-coordinate experiment, not a full trainer."""

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

    def parity_audit():
        before = restore()
        cam, gt = views['A'][0]
        _, before_image, _, _ = C.render_with_meta(before, cam, before.active_sh_degree, absgrad=False)
        C.fused_total_loss(before_image, gt).backward()
        parent_gradients = {g['name']: g['params'][0].grad[parent_ids].clone()
                            for g in before.optimizer.param_groups}
        del before, before_image
        physical, encoded = restore(), restore()
        hp, vp = split(physical, 20260926, 'native')
        he, ve = split(encoded, 20260926, 'pair_zero')
        assert hp == he
        props = {}
        for name in ('get_xyz', 'get_scaling', 'get_rotation', 'get_opacity', 'get_features'):
            a, b = getattr(vp, name), getattr(ve, name)
            assert torch.equal(a, b), name
            props[name] = True
        cam, gt = views['A'][0]
        _, ip, _, _ = C.render_with_meta(vp, cam, vp.active_sh_degree, absgrad=False)
        _, ie, _, _ = C.render_with_meta(ve, cam, ve.active_sh_degree, absgrad=False)
        image_max_error = float((ip-ie).abs().max())
        assert image_max_error == 0, image_max_error
        lp, le = C.fused_total_loss(ip, gt), C.fused_total_loss(ie, gt)
        assert float(lp) == float(le)
        lp.backward()
        le.backward()
        checks = {}
        parent_alignment = {}
        for pg, eg in zip(physical.optimizer.param_groups, encoded.optimizer.param_groups):
            gp, ge = pg['params'][0].grad, eg['params'][0].grad
            shared = ge[-2*nparents:-nparents].detach().flatten()
            parent = parent_gradients[pg['name']].flatten()
            parent_alignment[pg['name']] = {
                'shared_parent_cosine': float(torch.nn.functional.cosine_similarity(shared, parent, dim=0)),
                'shared_parent_norm_ratio': float(shared.norm()/parent.norm().clamp_min(1e-20))}
            expected = torch.cat((gp[:-2*nparents], gp[-2*nparents:-nparents]+gp[-nparents:],
                                  gp[-2*nparents:-nparents]-gp[-nparents:]))
            parts = {}
            for label, a, b in (('survivors', ge[:-2*nparents], expected[:-2*nparents]),
                                ('children', ge[-2*nparents:], expected[-2*nparents:])):
                absolute = float((a-b).abs().max())
                relative = float((a-b).norm() / b.norm().clamp_min(1e-20))
                assert relative < 1e-3 or absolute < 1e-8, (pg['name'], label, absolute, relative)
                parts[label] = {'max_absolute_error': absolute, 'relative_l2_error': relative}
            checks[pg['name']] = parts
        del physical, encoded, vp, ve, ip, ie, lp, le
        return {'exact_activated_properties': props, 'initial_image_max_error': image_max_error,
                'single_A_view_parent_alignment': parent_alignment,
                'gradient_chain_rule': checks, 'tolerance': 'relative L2 < 1e-3 OR max absolute < 1e-8; atomic reductions'}

    audit = parity_audit()
    print('PARITY PASS', args.scene, json.dumps(audit), flush=True)
    seeds = [20260926, 20260927, 20260928]
    policies = ('native', 'pair_zero', 'pair_shared', 'pair_full', 'inherit_05', 'inherit_02', 'physical_lr2')
    arms = [('no_edit', None, 'native')]
    for seed in seeds:
        for policy in policies:
            arms.append((f'{policy}_{seed}', seed, policy))
        arms.append((f'native_repeat_{seed}', seed, 'native'))
    arms.append(('no_edit_repeat', None, 'native'))
    result = {'scope': 'Single-birth shared/relative coordinate mechanism controls; no novelty or full-training claim',
              'scene': args.scene, 'N': n, 'nparents': nparents, 'checkpoint_sha256': sha(state_path),
              'candidate_source_sha256': sha(candidate_path), 'parent_ids': parent_ids.cpu().tolist(),
              'source_sha256': sha(__file__), 'update_helper_sha256': sha(root/'newborn_updates.py'),
              'coordinate_helper_sha256': sha(root/'sibling_coordinates.py'),
              'renderer_identity': identity, 'common_sha256': sha(C.__file__),
              'gaussian_model_sha256': sha(Path(C.REFERENCE_V1_DIR)/'gaussian_model.py'),
              'gpu': gpu, 'A_ids': a_ids, 'B_ids': b_ids, 'training_camera_sequence': training_ids,
              'old_group_steps': group_steps, 'parity_audit': audit,
              'protocol': {'split': 'native split; all arms canonicalized via shared/relative decode; fixed 256 parents',
                           'policies': list(policies), 'seeds': seeds, 'horizons': horizons,
                           'pair_zero': 'zero moments both modes; old clocks; native LR',
                           'pair_shared': 'full parent moments for shared mode; relative zero; both clocks old',
                           'pair_full': 'parent shared moments/clock; relative zero moments/fresh clock',
                           'inherit_05': 'physical child .5*m .25*v old clock',
                           'inherit_02': 'physical child .2*m .04*v old clock; NOT BOGausS reproduction',
                           'physical_lr2': 'physical children native update multiplied by 2; survivors native',
                           'B_seen_before': 'B excluded only from continuation; not formal unseen-view test',
                           'loss': '0.8 L1 + 0.2 fused D-SSIM, same padding, 1920x1080',
                           'frozen_later_topology': True, 'fixed_active_SH_degree': state['active_sh_degree'],
                           'overhead': 'Python full parameter decode and synchronized steps counted'},
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
        view_model = model
        if seed is not None:
            initial_hash, view_model = split(model, seed, policy)
            if seed in child_hashes:
                assert child_hashes[seed] == initial_hash, 'Policy changed split initialization'
            child_hashes[seed] = initial_hash
        torch.manual_seed(42)
        history = {'0': evaluate(view_model)}
        step_time, eval_time = 0.0, 0.0
        for step, camera_id in enumerate(training_ids, 1):
            step_begin = time.perf_counter()
            model.update_learning_rate(2000+step)
            cam, gt = dataset.get_item(camera_id)
            model.optimizer.zero_grad(set_to_none=True)
            _, image, _, _ = C.render_with_meta(view_model, cam, view_model.active_sh_degree, absgrad=False)
            loss = C.fused_total_loss(image, gt)
            loss.backward()
            update_policy = 'fresh_age' if policy == 'pair_full' else ('lr_scale' if policy == 'physical_lr2' else 'native')
            start_row = first_newborn+nparents if policy == 'pair_full' else first_newborn
            step_with_newborn_policy(model.optimizer, start_row if seed is not None else None,
                                     step, update_policy, 2.)
            model.optimizer.zero_grad(set_to_none=True)
            del image, loss
            torch.cuda.synchronize()
            step_time += time.perf_counter()-step_begin
            if step in horizons:
                evaluate_begin = time.perf_counter()
                history[str(step)] = evaluate(view_model)
                eval_time += time.perf_counter()-evaluate_begin
        result['arms'].append({'name': name, 'seed': seed, 'policy': policy,
                               'initial_children_sha256': initial_hash, 'history': history,
                               'train_seconds_with_sync': step_time, 'evaluation_seconds_after_initial': eval_time,
                               'arm_seconds': time.perf_counter()-begin})
        save()
        print(f'DONE {args.scene} {name} B@{args.steps}={history[str(args.steps)]["B"]["loss"]:.9f} train={step_time:.2f}s arm={time.perf_counter()-begin:.2f}s', flush=True)
        del view_model, model
    result['gpu_at_end'] = exclusive()
    result['completed'] = True
    save()
    print(f'COMPLETE {args.scene} {len(arms)} arms: {out}', flush=True)


if __name__ == '__main__':
    main()

"""Offline paired split experiment on faithful P11 states, with Adam history.

Run only in EPIC-05's pinned gsplat environment. No frozen source is modified.
This is a ranking/short-horizon diagnostic, not a deployed algorithm or speedup.
The repair basis consists of observed baseline Adam image secants, NOT a full
Jacobian. Fused original loss labels and sampled MSE proxy scores are separated.
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


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rank(x):
    x = np.asarray(x)
    order = np.argsort(x, kind='stable')
    result = np.empty(len(x), dtype=float)
    for value in np.unique(x):
        positions = np.flatnonzero(x[order] == value)
        result[order[positions]] = positions.mean()
    return result


def correlation(x, y):
    x, y = rank(x), rank(y)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='room')
    ap.add_argument('--iteration', type=int, default=2000)
    ap.add_argument('--gpu', type=int, default=5)
    ap.add_argument('--groups', type=int, default=8)
    ap.add_argument('--group-size', type=int, default=32)
    ap.add_argument('--steps', type=int, default=64)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.set_num_threads(4)
    sys.path.insert(0, '/root/epic05_p11')
    import p11_common as C
    gpu = C.assert_clean_gpu(args.gpu)
    # NVML reports host PIDs while /proc and os.getpid() here use container PIDs.
    # Attribute the sole new driver process across explicit CUDA initialization;
    # require exclusive ownership again before every arm and at completion.
    uuid = gpu['gpu_info'].split(',')[0].strip()
    torch.cuda.init()
    torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    driver_rows = [row for row in C.gpu_compute_apps() if uuid in row]
    assert len(driver_rows) == 1, f'Ambiguous CUDA context attribution: {driver_rows}'
    driver_pid = driver_rows[0].split(',')[0].strip()
    gpu.update({'namespace_pid': os.getpid(), 'driver_pid': driver_pid,
                'context_attribution': 'empty GPU before init; sole process after explicit own CUDA init'})

    def assert_exclusive():
        rows = [row for row in C.gpu_compute_apps() if uuid in row]
        assert {row.split(',')[0].strip() for row in rows} == {driver_pid}, f'GPU ownership changed: {rows}'
        return rows
    identity = C.bootstrap_gmain()
    C.import_recipe()
    from gaussian_model import GaussianModel

    started = time.perf_counter()
    state_path = Path(C.STATEGEN_DIR) / args.scene / f'state_{args.iteration}.pt'
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    assert state['iteration'] == args.iteration
    assert state['optimizer_state_dict']['state'], 'Missing Adam history'
    dataset = C.load_dataset(args.scene)
    extent = state['spatial_lr_scale']
    seq_path = state_path.with_name('camera_sequence.npy')
    sequence = np.load(seq_path).tolist()
    unique = list(dict.fromkeys(sequence[args.iteration:]))
    fit_ids, val_ids = unique[:3], unique[3:6]
    assert len(val_ids) == 3 and not set(fit_ids) & set(val_ids)
    # Exclude B from all diagnostic continuation updates.
    training_ids = [x for x in sequence[args.iteration:] if x not in val_ids][:args.steps]
    assert len(training_ids) == args.steps
    fit_views = [dataset.get_item(i) for i in fit_ids]
    val_views = [dataset.get_item(i) for i in val_ids]
    target_samples = np.concatenate([gt[::8, ::8].cpu().numpy().reshape(-1) for _, gt in fit_views]).astype(np.float64)
    horizons = sorted(set([0, 8, 32, args.steps]))
    assert args.steps >= 32
    optimizer_checks = []

    def restore():
        model = GaussianModel(max_sh_degree=3)
        model.restore(state, C.TRAIN_CFG)
        assert len(model.optimizer.state) == 0, 'Audit restore semantics changed'
        # restore() does NOT load this, despite it being present in the file.
        model.optimizer.load_state_dict(copy.deepcopy(state['optimizer_state_dict']))
        model.xyz_gradient_accum = state['xyz_gradient_accum'].cuda().clone()
        model.denom = state['denom'].cuda().clone()
        for group, saved_group in zip(model.optimizer.param_groups, state['optimizer_state_dict']['param_groups']):
            actual = model.optimizer.state[group['params'][0]]
            saved = state['optimizer_state_dict']['state'][saved_group['params'][0]]
            assert float(actual['step']) == float(saved['step'])
            for key in ('exp_avg', 'exp_avg_sq'):
                assert torch.equal(actual[key].reshape(-1)[:4096].cpu(), saved[key].reshape(-1)[:4096])
        return model

    @torch.no_grad()
    def evaluate(model, sample=False):
        scores, samples = {}, []
        for name, views in [('A', fit_views), ('B', val_views)]:
            losses, mses = [], []
            for cam, gt in views:
                _, image, _, _ = C.render_with_meta(model, cam, model.active_sh_degree, absgrad=False)
                losses.append(float(C.fused_total_loss(image, gt)))
                mses.append(float((image - gt).square().mean()))
                if name == 'A' and sample:
                    samples.append(image[::8, ::8].cpu().numpy().reshape(-1))
            scores[name] = {'loss': float(np.mean(losses)), 'mse': float(np.mean(mses)),
                            'mean_view_psnr': float(np.mean(-10 * np.log10(mses)))}
        return scores, np.concatenate(samples).astype(np.float64) if sample else None

    @torch.no_grad()
    def sample_A(model):
        samples = []
        for cam, _ in fit_views:
            _, image, _, _ = C.render_with_meta(model, cam, model.active_sh_degree, absgrad=False)
            samples.append(image[::8, ::8].cpu().numpy().reshape(-1))
        return np.concatenate(samples).astype(np.float64)

    @torch.no_grad()
    def local_repair_columns(model, ids):
        # Real parameter-space central differences. Ten shared group directions,
        # not masked image-space responses and not a full per-Gaussian Jacobian.
        spatial_step = 0.01 * float(model.get_scaling[ids].median())
        directions = [(model._xyz, (axis,), spatial_step) for axis in range(3)]
        directions += [(model._scaling, (axis,), 0.01) for axis in range(3)]
        directions += [(model._opacity, (), 0.01)]
        directions += [(model._shs, (0, channel), 0.01) for channel in range(3)]
        columns = []
        for parameter, coordinate, step in directions:
            old = parameter[ids].clone()
            key = (ids,) + coordinate
            parameter[key] += step
            plus = sample_A(model)
            parameter[ids] = old
            parameter[key] -= step
            minus = sample_A(model)
            parameter[ids] = old
            assert torch.equal(parameter[ids], old)
            columns.append((plus - minus) / (2 * step))
        return np.stack(columns, axis=1)

    def orthonormal_basis(j):
        norms = np.linalg.norm(j, axis=0)
        j = j[:, norms > 1e-20] / norms[norms > 1e-20]
        u, singular, _ = np.linalg.svd(j, full_matrices=False)
        return u[:, singular > singular[0] * 1e-6], singular

    # Candidate generation uses only A and a fixed 4x4 screen partition.
    model = restore()
    n = state['num_points']
    grad_score = torch.zeros(n, device='cuda')
    seen = torch.zeros(n, device='cuda', dtype=torch.int32)
    first_xy, first_radii, error_proxy = None, None, None
    for view_index, (cam, gt) in enumerate(fit_views):
        model.optimizer.zero_grad(set_to_none=True)
        _, image, radii, means2d = C.render_with_meta(model, cam, model.active_sh_degree)
        loss = C.fused_total_loss(image, gt)
        loss.backward()
        g = means2d.absgrad[0].detach().clone()
        g[:, 0] *= cam.image_width / 2.0
        g[:, 1] *= cam.image_height / 2.0
        visible = (radii[0] > 0).any(-1)
        grad_score += g.norm(dim=-1) * visible
        seen += visible
        if view_index == 0:
            first_xy = means2d[0].detach().clone()
            first_radii = radii[0].detach().clone()
            # Exact rectangle integral of high-error pixels: a footprint proxy,
            # NOT official FastGS VCD (only one view; rectangular footprint).
            high = ((image.detach() - gt).abs().mean(-1) > 0.02).double()
            sat = torch.nn.functional.pad(high.cumsum(0).cumsum(1), (1, 0, 1, 0))
            lo = torch.floor(first_xy - first_radii).long()
            hi = torch.ceil(first_xy + first_radii).long() + 1
            x0, x1 = lo[:, 0].clamp(0, cam.image_width), hi[:, 0].clamp(0, cam.image_width)
            y0, y1 = lo[:, 1].clamp(0, cam.image_height), hi[:, 1].clamp(0, cam.image_height)
            error_proxy = (sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0]).float()
        del image, loss, means2d, radii
    grad_score /= seen.clamp_min(1)
    cam0 = fit_views[0][0]
    eligible = (seen >= 2) & (first_radii > 0).all(-1)
    eligible &= model.get_scaling.max(-1).values > model.percent_dense * extent
    eligible &= (first_xy[:, 0] >= 0) & (first_xy[:, 0] < cam0.image_width)
    eligible &= (first_xy[:, 1] >= 0) & (first_xy[:, 1] < cam0.image_height)
    cell = (first_xy[:, 0] * 4 / cam0.image_width).long().clamp(0, 3)
    cell += 4 * (first_xy[:, 1] * 4 / cam0.image_height).long().clamp(0, 3)
    candidates = []
    for tile in range(16):
        pool = torch.nonzero(eligible & (cell == tile)).flatten()
        if len(pool) < args.group_size:
            continue
        ids = pool[torch.topk(grad_score[pool], args.group_size).indices]
        candidates.append({'cell': tile, 'ids': ids.cpu().tolist(),
                           'absgrad': float(grad_score[ids].mean()),
                           'footprint_proxy': float(error_proxy[ids].mean())})
    candidates.sort(key=lambda x: -x['absgrad'])
    candidates = candidates[:args.groups]
    assert len(candidates) >= 4, f'Only {len(candidates)} valid candidate groups'
    initial, initial_samples = evaluate(model, sample=True)
    del model
    residual = initial_samples - target_samples
    print(f'START {args.scene}@{args.iteration} N={n} candidates={len(candidates)} A={fit_ids} B={val_ids}', flush=True)

    result = {'scope': 'Full-resolution real-state offline paired split ranking; no E2E speedup claim',
              'scene': args.scene, 'iteration': args.iteration, 'N': n,
              'gpu': gpu, 'renderer_identity': identity, 'environment': C.env_identity(),
              'source_sha256': sha(__file__), 'checkpoint_sha256': sha(state_path),
              'common_source_sha256': sha(C.__file__),
              'gaussian_model_sha256': sha(Path(C.REFERENCE_V1_DIR) / 'gaussian_model.py'),
              'camera_sequence_sha256': sha(seq_path), 'A': fit_ids, 'B': val_ids,
              'training_camera_sequence': training_ids, 'initial': initial,
              'protocol': {'loss': '0.8 L1 + 0.2 fused D-SSIM, same padding',
                           'diagnostic_topology_frozen_after_single_split': True,
                           'no_opacity_reset_or_sh_upgrades_within_short_window': True,
                           'B_excluded_from_continuation': True,
                           'split': 'native model.densify_and_split, N=2, unchanged initialization',
                           'repair_basis': '3 full-Adam image secants at 1,4,8 steps, and 10 local parameter central differences; compared separately and combined',
                           'score_pixels': 'A every eighth pixel per axis, all RGB',
                           'official_FastGS_or_SteepGS_comparison': False,
                           'dataset_split': 'P11 recipe camera list; B held out only from continuation, not original checkpoint training'},
              'data_sources': {'camera_json_sha256': sha(dataset.official_dir / 'cameras.json'),
                               'dataset_source_sha256': sha(Path(C.PHASE7_DIR) / 'dataset.py'),
                               'images': {str(i): {'path': str(dataset._valid_image_paths[i]), 'sha256': sha(dataset._valid_image_paths[i])}
                                          for i in sorted(set(training_ids + fit_ids + val_ids))}},
              'arms': [], 'candidates': candidates}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        result['elapsed_seconds_including_setup'] = time.perf_counter() - started
        tmp = out.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, indent=2) + '\n')
        tmp.replace(out)

    repair_samples = []
    basis = None
    arms = [('base', None), ('base_repeat', None)] + [(f'edit_{i}', a) for i, a in enumerate(candidates)]
    for name, candidate in arms:
        assert_exclusive()
        arm_start = time.perf_counter()
        model = restore()
        torch.manual_seed(20260926)
        if candidate is not None:
            ids = torch.tensor(candidate['ids'], device='cuda')
            local_started = time.perf_counter()
            local_j = local_repair_columns(model, ids)
            local_basis, local_singular = orthonormal_basis(local_j)
            combined_basis, _ = orthonormal_basis(np.column_stack([local_j, *repair_samples]))
            local_seconds = time.perf_counter() - local_started
            mask = torch.ones(n, device='cuda', dtype=torch.bool)
            mask[ids] = False
            old_states = {g['name']: {k: v.clone() if torch.is_tensor(v) else v
                                    for k, v in model.optimizer.state[g['params'][0]].items()}
                          for g in model.optimizer.param_groups}
            gradients = torch.zeros(n, device='cuda')
            gradients[ids] = 1
            with torch.no_grad():
                count = model.densify_and_split(gradients, 0.5, extent, N=2)
            assert count == len(ids) and len(model._xyz) == n + len(ids)
            for g in model.optimizer.param_groups:
                current = model.optimizer.state[g['params'][0]]
                old = old_states[g['name']]
                assert float(current['step']) == float(old['step'])
                for key in ('exp_avg', 'exp_avg_sq'):
                    assert torch.equal(current[key][:n-len(ids)], old[key][mask])
                    assert torch.count_nonzero(current[key][n-len(ids):]) == 0
            del old_states
            optimizer_checks.append(name)
        # Identical continuation RNG after topology generation, both arms.
        torch.manual_seed(42)
        before, samples = evaluate(model, sample=candidate is not None)
        arm = {'name': name, 'history': {'0': before}, 'N': len(model._xyz)}
        if candidate is not None:
            b = samples - initial_samples
            pr = residual - basis @ (basis.T @ residual)
            pb = b - basis @ (basis.T @ b)
            arm['scores'] = {
                'absgrad': candidate['absgrad'], 'footprint_proxy': candidate['footprint_proxy'],
                'raw_finite_mse': float((-b @ residual - 0.5 * b @ b) / len(b)),
                'conditional_secant_mse': float((-b @ pr - 0.5 * b @ pb) / len(b)),
                'raw_finite_original_loss_A': initial['A']['loss'] - before['A']['loss'],
                'edit_fraction_explained': float(1 - (pb @ pb) / max(b @ b, 1e-30))}
            for label, q in [('local', local_basis), ('combined', combined_basis)]:
                prq = residual - q @ (q.T @ residual)
                pbq = b - q @ (q.T @ b)
                arm['scores'][f'conditional_{label}_mse'] = float((-b @ prq - 0.5 * b @ pbq) / len(b))
                arm['scores'][f'{label}_edit_fraction_explained'] = float(1 - (pbq @ pbq) / max(b @ b, 1e-30))
            arm['local_basis_seconds'] = local_seconds
            arm['local_basis_rank'] = local_basis.shape[1]
            arm['local_basis_singular_values'] = local_singular.tolist()
        train_start = time.perf_counter()
        for step, camera_id in enumerate(training_ids, 1):
            model.update_learning_rate(args.iteration + step)
            cam, gt = dataset.get_item(camera_id)
            model.optimizer.zero_grad(set_to_none=True)
            _, image, _, _ = C.render_with_meta(model, cam, model.active_sh_degree, absgrad=False)
            loss = C.fused_total_loss(image, gt)
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
            del image, loss
            if step in horizons or (name == 'base' and step in (1, 4)):
                metrics, samp = evaluate(model, sample=name == 'base' and step in (1, 4, 8))
                if step in horizons:
                    arm['history'][str(step)] = metrics
                if samp is not None:
                    repair_samples.append(samp - initial_samples)
        torch.cuda.synchronize()
        arm['continuation_and_evaluation_seconds'] = time.perf_counter() - train_start
        arm['arm_seconds_including_restore_edit_scoring'] = time.perf_counter() - arm_start
        result['arms'].append(arm)
        if name == 'base':
            j = np.stack(repair_samples, axis=1)
            u, singular, _ = np.linalg.svd(j, full_matrices=False)
            basis = u[:, singular > singular[0] * 1e-6]
            result['repair_basis_singular_values'] = singular.tolist()
            result['repair_basis_rank'] = basis.shape[1]
        del model
        save()
        print(f'DONE {name} lossB@{args.steps}={arm["history"][str(args.steps)]["B"]["loss"]:.9f} seconds={arm["arm_seconds_including_restore_edit_scoring"]:.2f}', flush=True)

    result['optimizer_survivor_and_newborn_checks_passed'] = optimizer_checks
    result['analysis'] = {}
    for h in horizons:
        base_loss = result['arms'][0]['history'][str(h)]['B']['loss']
        repeat_loss = result['arms'][1]['history'][str(h)]['B']['loss']
        edits = result['arms'][2:]
        gains = [base_loss - x['history'][str(h)]['B']['loss'] for x in edits]
        noise = abs(base_loss - repeat_loss)
        ranking = {}
        for score in ('absgrad', 'footprint_proxy', 'raw_finite_mse', 'conditional_secant_mse', 'conditional_local_mse', 'conditional_combined_mse', 'raw_finite_original_loss_A'):
            values = [x['scores'][score] for x in edits]
            top = np.argsort(values)[-min(3, len(edits)):]
            ranking[score] = {'spearman': correlation(values, gains),
                              'top3_mean_gain': float(np.mean(np.array(gains)[top])),
                              'top3_indices': top.tolist()}
        result['analysis'][str(h)] = {'B_loss_gains': gains, 'base_repeat_absolute_difference': noise,
                                     'positive_gain_exceeding_repeat_difference': int(np.sum(np.array(gains) > noise)),
                                     'rankings': ranking}
    result['gpu_at_completion'] = assert_exclusive()
    result['completed'] = True
    save()
    print(json.dumps(result['analysis'], indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Bound the native fixed-batch covariance/opacity chain-rule reuse opportunity."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch


REPO = Path('/root/codex_fastgs_audit_20260927')
os.chdir(REPO)
sys.path.insert(0, str(REPO))
from fused_ssim import fused_ssim
from gaussian_renderer import render_fastgs
from scene import GaussianModel, Scene, dataset_readers
from utils.loss_utils import l1_loss


PARAMETERS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')


def sha256(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(data)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', choices=('room', 'garden'), default='room')
    args = parser.parse_args()
    if args.scene == 'room':
        model_name, iteration, expected = ('room_big_branch_base30k', 30000,
            'c2afc615fd244d561bd790e7d7721af1b7a7515222bd5385650e7b886c58516a')
    else:
        model_name, iteration, expected = ('garden_exchange_dev_4100', 4100,
            'd396092ca139fa0e6aea4d17a085ca0239bf2ff0ec2bbbe0ee3c5ab539feb5fb')
    ply = REPO / 'output' / model_name / 'point_cloud' / f'iteration_{iteration}/point_cloud.ply'
    assert sha256(ply) == expected
    dataset = SimpleNamespace(
        source_path='/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/' + args.scene,
        model_path=str(REPO / 'output' / model_name), images='images', resolution=-1,
        white_background=False, data_device='cuda', eval=False)
    original_reader = dataset_readers.readColmapCameras
    chosen_names = []

    def selected_reader(cam_extrinsics, cam_intrinsics, images_folder):
        sorted_cameras = sorted(cam_extrinsics.items(), key=lambda item: Path(item[1].name).stem)
        train = [item for i, item in enumerate(sorted_cameras) if i % 8 != 0]
        indices = np.random.default_rng(20260927).choice(len(train), 64, replace=False)
        chosen = [train[int(i)] for i in indices]
        chosen_names.extend(Path(item[1].name).stem for item in chosen)
        return original_reader(dict(chosen), cam_intrinsics, images_folder)

    model = GaussianModel(3, 'default')
    dataset_readers.readColmapCameras = selected_reader
    try:
        scene = Scene(dataset, model, load_iteration=iteration, shuffle=False)
    finally:
        dataset_readers.readColmapCameras = original_reader
    by_name = {camera.image_name: camera for camera in scene.getTrainCameras()}
    cameras = [by_name[name] for name in chosen_names]
    assert len(cameras) == len(set(chosen_names)) == 64 and not scene.getTestCameras()
    assert all(camera.image_width == 1600 for camera in cameras)
    background = torch.zeros(3, device='cuda')
    native_pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    deferred_pipe = SimpleNamespace(debug=False, compute_cov3D_python=True, convert_SHs_python=False)
    original_covariance, original_opacity = model.get_covariance, model.opacity_activation

    def block(mode, views, capture=False):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for name in PARAMETERS:
            getattr(model, name).grad = None
        if mode == 'deferred':
            covariance_graph = original_covariance()
            opacity_graph = original_opacity(model._opacity)
            covariance = covariance_graph.detach().requires_grad_(True)
            opacity = opacity_graph.detach().requires_grad_(True)
            model.get_covariance = lambda scaling_modifier=1.: covariance
            model.opacity_activation = lambda raw: opacity
        images = []
        try:
            for camera in views:
                image = render_fastgs(camera, model,
                    deferred_pipe if mode == 'deferred' else native_pipe, background, .5)['render']
                gt = camera.original_image
                loss = .8 * l1_loss(image, gt) + .2 * (1 - fused_ssim(image[None], gt[None]))
                loss.backward()
                # Match the author's per-view host synchronization in either arm.
                loss.item()
                torch.cuda.synchronize()
                if capture:
                    images.append(image.detach().clone())
            if mode == 'deferred':
                torch.autograd.backward((covariance_graph, opacity_graph),
                                        (covariance.grad, opacity.grad))
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            gradients = {name: getattr(model, name).grad.detach().clone()
                         for name in PARAMETERS} if capture else None
            return seconds, images, gradients
        finally:
            model.get_covariance = original_covariance
            model.opacity_activation = original_opacity

    result = {'scene': args.scene, 'model': model_name, 'iteration': iteration,
              'ply_sha256': expected, 'source_commit': '44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f',
              'probe_sha256': sha256(Path(__file__)), 'gaussians': len(model.get_xyz),
              'camera_names': chosen_names, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'scope': 'Frozen-state 64 distinct train-view forward/loss/backward blocks. Includes cache construction and deferred pullback; excludes optimizer and all full-training lifecycle costs.'}
    output = REPO / f'fastgs_deferred_pullback_{args.scene}_result.json'

    def save():
        output.write_text(json.dumps(result, indent=2) + '\n')

    _, native_images, native_gradients = block('native', cameras[:4], True)
    parity = {}
    for mode in ('native', 'deferred'):
        _, images, gradients = block(mode, cameras[:4], True)
        errors = {}
        for name in PARAMETERS:
            reference, value = native_gradients[name], gradients[name]
            difference = value - reference
            absolute = float(difference.abs().max())
            relative = float(torch.linalg.vector_norm(difference.double()) /
                             torch.linalg.vector_norm(reference.double()).clamp_min(1e-30))
            errors[name] = {'relative_l2': relative, 'max_abs': absolute,
                            'pass': relative <= 1e-3 or absolute <= 1e-7}
        rgb_error = max(float((a - b).abs().max()) for a, b in zip(images, native_images))
        parity[mode] = {'max_rgb_error': rgb_error, 'gradients': errors,
                        'pass': rgb_error <= 1e-4 and all(row['pass'] for row in errors.values())}
        del images, gradients
    result['parity'] = parity
    save()
    del native_images, native_gradients
    if not all(row['pass'] for row in parity.values()):
        result['decision'] = 'stop_parity'
        save()
        print(json.dumps(result, indent=2), flush=True)
        return
    for mode in ('native', 'deferred'):
        block(mode, cameras)
    pairs = []
    for index in range(5):
        row = {}
        for mode in (('native', 'deferred') if index % 2 == 0 else ('deferred', 'native')):
            row[mode] = block(mode, cameras)[0]
        row['ratio'] = row['deferred'] / row['native']
        pairs.append(row)
    result['pairs'] = pairs
    result['median_wall_ratio'] = float(np.median([row['ratio'] for row in pairs]))
    result['decision'] = ('cost_gate_pass_only' if result['median_wall_ratio'] <= .85
                          and all(row['ratio'] < 1 for row in pairs) else 'stop_cost')
    assert sha256(ply) == expected
    save()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

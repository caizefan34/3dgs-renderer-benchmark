#!/usr/bin/env python3
"""Status: IMPLEMENTED; real MX fixtures NOT_EXECUTED.

Read the exact R18 checkpoint schema and 3DGS cameras.json convention.
Checkpoint and camera files are read-only and SHA256-checked before/after.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from R20_GRADIENT_PARITY_TESTS import FIELDS, sha, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--cameras-json', required=True); p.add_argument('--camera-ids', default='0')
    p.add_argument('--width', type=int, default=960); p.add_argument('--height', type=int, default=540)
    p.add_argument('--subset', type=int, default=0, help='0 keeps all Gaussian rows; otherwise seed-42 sorted subset')
    p.add_argument('--extent', type=float, required=True, help='identical reference_v1 training scene extent')
    p.add_argument('--max-screen-size', type=int, default=20); p.add_argument('--output', required=True)
    a = p.parse_args()
    checkpoint_sha, camera_sha = sha(a.checkpoint), sha(a.cameras_json)
    if checkpoint_sha != a.checkpoint_sha256: raise RuntimeError('checkpoint SHA256 mismatch')
    # Existing project checkpoints use this pickle schema (R18 loader).
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    for k in FIELDS:
        if k not in ckpt: raise RuntimeError('R18 checkpoint field missing: '+k)
    n = len(ckpt['xyz'])
    ids = np.arange(n) if a.subset == 0 else np.sort(np.random.RandomState(42).choice(n, a.subset, replace=False))
    fixture = {k: ckpt[k][ids].float().contiguous() for k in FIELDS}
    fixture['opacity'] = fixture['opacity'].reshape(-1)
    cameras = json.loads(Path(a.cameras_json).read_text(encoding='utf-8'))
    if not isinstance(cameras, list): raise RuntimeError('3DGS cameras.json must be a list')
    selected = [int(x) for x in a.camera_ids.split(',')]
    views, ks = [], []
    for cid in selected:
        c = cameras[cid]; world = np.eye(4, dtype=np.float32)
        world[:3, :3] = np.asarray(c['rotation'], dtype=np.float32)
        world[:3, 3] = np.asarray(c['position'], dtype=np.float32)
        views.append(torch.from_numpy(np.linalg.inv(world).astype(np.float32)))
        ks.append(torch.tensor([[a.width*float(c['fx'])/c['width'], 0., a.width/2.],
                                [0., a.height*float(c['fy'])/c['height'], a.height/2.], [0., 0., 1.]]))
    c = len(selected); generator = torch.Generator().manual_seed(42)
    degree = round(fixture['shs'].shape[1]**.5)-1
    if (degree+1)**2 != fixture['shs'].shape[1]: raise RuntimeError('invalid SH coefficient count')
    fixture.update(viewmats=torch.stack(views), Ks=torch.stack(ks), width=a.width, height=a.height,
        sh_degree=degree, eps2d=.1, extent=a.extent, max_screen_size=a.max_screen_size,
        v_rgb=torch.randn(c,a.height,a.width,3,generator=generator)/(c*a.width*a.height*3),
        v_alpha=torch.randn(c,a.height,a.width,1,generator=generator)*.1/(c*a.width*a.height))
    output = Path(a.output).resolve()
    if output in (Path(a.checkpoint).resolve(), Path(a.cameras_json).resolve()) or output.exists():
        raise RuntimeError('fixture output must be a new file')
    output.parent.mkdir(parents=True, exist_ok=True); torch.save(fixture, output)
    assert sha(a.checkpoint) == checkpoint_sha and sha(a.cameras_json) == camera_sha
    write_json(output.with_suffix('.json'), dict(status='FIXTURE_CREATED_GPU_NOT_EXECUTED',
        checkpoint=a.checkpoint, checkpoint_sha256=checkpoint_sha, cameras_json=a.cameras_json,
        cameras_sha256=camera_sha, camera_ids=selected, n=len(ids), fixture_sha256=sha(output),
        extent=a.extent, gaussian_indices_sha256=__import__('hashlib').sha256(ids.astype('<i8').tobytes()).hexdigest()))


if __name__ == '__main__':
    main()

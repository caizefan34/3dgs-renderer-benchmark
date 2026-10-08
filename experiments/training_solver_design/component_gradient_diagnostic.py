"""Isolate native/component discrepancies from loss-adjoint differences."""

import argparse
import copy
import json
import os
import random
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from multilevel_loss import block_means, fused_loss_map, corrected_loss
from masked_renderer import render_masked, core_tile_mask
from real_topology_probe import sha


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
    result = {'scope': 'Sparse halo correctness and fixed-state work cost including zero-LR Adam; no training quality claim',
              'scene': args.scene, 'N': saved['num_points'], 'source_sha256': sha(__file__),
              'helper_sha256': sha(root/'multilevel_loss.py'), 'masked_helper_sha256': sha(root/'masked_renderer.py'),
              'checkpoint_sha256': sha(state_path), 'renderer_identity': identity,
              'common_sha256': sha(C.__file__), 'gpu': gpu,
              'protocol': {'views': previous['A'], 'block_grid': [9,15], 'coarse_factor': 2,
                           'halo': 5, 'sampling_fraction': .25,
                           'timing': 'fixed parameters; LR=0 Adam still updates moments; 3 randomized rounds of 10 steps, 4 warmups',
                           'included': 'full projection/intersections both resolutions, dense loss maps, sparse fine raster, optimizer and Python dispatch',
                           'excluded': 'mask construction/GT upload precomputed for each fixed view; no densification or training progression',
                           'gradient_gate': 'relative L2 < .003 OR max abs < 1e-6; native GPU reductions are nondeterministic'},
              'records': []}

    def save():
        result['wall_seconds'] = time.perf_counter()-started
        Path(args.out).write_text(json.dumps(result, indent=2)+'\n')

    def discrepancy(a, b):
        return {'relative_l2': float(((a-b).norm()/b.norm().clamp_min(1e-20)).detach()),
                'max_abs': float((a-b).abs().max().detach())}

    for camera_id in previous['A']:
        exclusive()
        cam,gt=dataset.get_item(camera_id)
        _,native,_,_=C.render_with_meta(model,cam,model.active_sh_degree,absgrad=False)
        other=render_masked(model,cam,eps2d=C.EPS2D)
        _,repeat,_,_=C.render_with_meta(model,cam,model.active_sh_degree,absgrad=False)
        adjoint,=torch.autograd.grad(C.fused_total_loss(native,gt),native,retain_graph=True)
        gn=torch.autograd.grad(C.fused_total_loss(native,gt),params,retain_graph=True)
        gr=torch.autograd.grad(C.fused_total_loss(repeat,gt),params)
        go=torch.autograd.grad(C.fused_total_loss(other,gt),params,retain_graph=True)
        ge=torch.autograd.grad(other,params,grad_outputs=adjoint)
        record={'camera_id':camera_id,'image_error':discrepancy(other,native),
                'native_repeat_image_exact':torch.equal(native,repeat),
                'L1_sign_changed_components':int(((native-gt).sign()!=(other-gt).sign()).sum()),
                'different_clamped_endpoints':int(((native==0)!=(other==0)).sum()+int(((native==1)!=(other==1)).sum())),
                'groups':{group['name']:{'native_repeat':discrepancy(r,n),'component_own_adjoint':discrepancy(o,n),
                                       'component_shared_adjoint':discrepancy(e,n)}
                          for group,n,r,o,e in zip(groups,gn,gr,go,ge)}}
        result['records'].append(record);save()
        print('DIAGNOSTIC',args.scene,json.dumps(record),flush=True)
        del native,other,repeat,adjoint,gn,gr,go,ge
    result['scope']='Native/component gradient diagnostic with common image adjoint; not performance or training'
    result['completed']=True;result['gpu_at_end']=exclusive();save()
    print('COMPLETE',args.scene,args.out,flush=True)


if __name__=='__main__':
    main()

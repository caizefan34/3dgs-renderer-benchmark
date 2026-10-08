"""Sparse halo correctness and fixed-state work cost, not training speedup."""

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
from masked_renderer_v2 import render_masked, core_tile_mask
from real_topology_probe import sha
from sort_reuse_renderer import render_shared, prepare, intersections
from patch_loss import extract_patches, patch_weighted_loss


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
    result = {'scope': 'Fixed-state cached coarse-list counterfactual and fine16/coarse8 list identity; no usable training cache or speed claim',
              'scene': args.scene, 'N': saved['num_points'], 'source_sha256': sha(__file__),
              'helper_sha256': sha(root/'multilevel_loss.py'), 'masked_helper_sha256': sha(root/'masked_renderer_v2.py'), 'shared_helper_sha256': sha(root/'sort_reuse_renderer.py'), 'patch_helper_sha256': sha(root/'patch_loss.py'),
              'checkpoint_sha256': sha(state_path), 'renderer_identity': identity,
              'common_sha256': sha(C.__file__), 'gpu': gpu,
              'protocol': {'views': previous['A'], 'block_grid': [9,15], 'coarse_factor': 2,
                           'halo': 5, 'sampling_fraction': .25,
                           'timing': 'fixed parameters; LR=0 Adam still updates moments; 3 randomized rounds of 10 steps, 4 warmups',
                           'included': 'dense loss maps, intersections at both resolutions, optimizer and Python dispatch; shared mode reuses fine projection/SH, proxy visibility and pixel filter come from fine',
                           'excluded': 'mask construction, target patches and GT upload precomputed for each fixed view; no densification or training progression',
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
        cam, gt = dataset.get_item(camera_id)
        h,w = cam.image_height,cam.image_width
        assert (h,w)==(1080,1920)
        generator = torch.Generator().manual_seed(701)
        labels = (torch.randperm(135,generator=generator).reshape(9,15)%4).cuda()
        torch.cuda.synchronize()
        mask_start = time.perf_counter()
        masks = [core_tile_mask(labels,i,h,w) for i in range(4)]
        torch.cuda.synchronize()
        mask_seconds = time.perf_counter()-mask_start
        weights = [(labels==i).float()*4 for i in range(4)]
        selected = [torch.where(labels==i) for i in range(4)]
        target_patches = [extract_patches(gt,*indices) for indices in selected]
        K=cam.K.clone();K[:2]/=2
        proxy_cam=SimpleNamespace(K=K,viewmatrix=cam.viewmatrix,image_width=w//2,image_height=h//2)
        proxy_gt=F.interpolate(gt.permute(2,0,1)[None],size=(h//2,w//2),mode='area')[0].permute(1,2,0).contiguous()

        with torch.no_grad():
            prepared = prepare(model,cam,C.EPS2D)
            fine_list = intersections(prepared,w,h)
            scaled8_list = intersections(prepared,w//2,h//2,tile=8,coarse=True)
            cached16_list = intersections(prepared,w//2,h//2,coarse=True)
            list_checks = {key: {'equal': torch.equal(a,b), 'fine_numel': a.numel(), 'coarse_numel': b.numel()}
                           for key,a,b in zip(('counts','ids','flat','offsets'),fine_list,scaled8_list)}
            stage_timing = {}
            for name, tile, coarse in [('fine16',16,False),('coarse16',16,True),('coarse8',8,True)]:
                width,height=(w//2,h//2) if coarse else (w,h)
                for _ in range(4): intersections(prepared,width,height,tile,coarse)
                values=[]
                for repeat in range(3):
                    torch.cuda.synchronize()
                    timer=time.perf_counter()
                    for _ in range(20): intersections(prepared,width,height,tile,coarse)
                    torch.cuda.synchronize()
                    values.append((time.perf_counter()-timer)*50)
                stage_timing[name]=values
            del prepared,fine_list,scaled8_list

        _, original, _, _ = C.render_with_meta(model,cam,model.active_sh_degree,absgrad=False)
        dense, unused_proxy=render_shared(model,cam,eps2d=C.EPS2D)
        del unused_proxy
        image_error=discrepancy(dense,original)
        assert image_error['max_abs']<5e-6, image_error
        native_grad=torch.autograd.grad(C.fused_total_loss(original,gt),params)
        dense_grad=torch.autograd.grad(C.fused_total_loss(dense,gt),params,retain_graph=True)
        native_parity={group['name']:discrepancy(a,b) for group,a,b in zip(groups,dense_grad,native_grad)}
        for v in native_parity.values():
            assert v['relative_l2']<.003 or v['max_abs']<1e-6,v
        del native_grad,dense_grad,original
        dense_blocks=block_means(fused_loss_map(dense,gt),9,15)
        halo_checks=[]
        for label,(core,covered,mask) in enumerate(masks):
            sparse, unused_proxy=render_shared(model,cam,mask,eps2d=C.EPS2D)
            del unused_proxy
            assert torch.equal(sparse[covered],dense[covered]), 'Masked rendering changed covered pixels'
            sparse_blocks=block_means(fused_loss_map(sparse,gt),9,15)
            loss=patch_weighted_loss(sparse,target_patches[label],*selected[label])
            reference=(weights[label]*dense_blocks).mean()
            loss_error=abs(float((loss-reference).detach()))
            assert loss_error<5e-7,loss_error
            ga=torch.autograd.grad(loss,params)
            gb=torch.autograd.grad(reference,params,retain_graph=True)
            gp={group['name']:discrepancy(a,b) for group,a,b in zip(groups,ga,gb)}
            for v in gp.values():
                assert v['relative_l2']<.003 or v['max_abs']<1e-6,v
            halo_checks.append({'label':label,'exact_covered_pixels':True,'loss_error':loss_error,
                                'gradient_parity':gp,'active_tile_fraction':float(mask.float().mean()),
                                'core_fraction':float(core.float().mean())})
            del sparse,sparse_blocks,loss,reference,ga,gb
        # Negative control: omitting halo changes the original SSIM objective.
        _,_,nohalo=core_tile_mask(labels,0,h,w,halo=0)
        wrong=render_masked(model,cam,nohalo,eps2d=C.EPS2D)
        wrong_loss=(weights[0]*block_means(fused_loss_map(wrong,gt),9,15)).mean()
        missing_halo_error=abs(float((wrong_loss-(weights[0]*dense_blocks).mean()).detach()))
        del dense,dense_blocks,wrong,wrong_loss

        def step(mode,iteration):
            model.optimizer.zero_grad(set_to_none=True)
            label=iteration%4
            if mode=='native_dense':
                _,image,_,_=C.render_with_meta(model,cam,model.active_sh_degree,absgrad=False)
                loss=C.fused_total_loss(image,gt)
            elif mode=='components_dense':
                image=render_masked(model,cam,eps2d=C.EPS2D)
                loss=C.fused_total_loss(image,gt)
            else:
                if mode in ('shared','shared_patch','free_coarse_sort'):
                    image,proxy=render_shared(model,cam,masks[label][2],eps2d=C.EPS2D,
                                              coarse_list=cached16_list if mode=='free_coarse_sort' else None)
                else:
                    image=render_masked(model,cam,masks[label][2],eps2d=C.EPS2D)
                if mode not in ('shared_patch','free_coarse_sort'):
                    blocks=block_means(fused_loss_map(image,gt),9,15)
                if mode=='uniform_sparse':
                    loss=(weights[label]*blocks).mean()
                else:
                    if mode not in ('shared','shared_patch','free_coarse_sort'):
                        _,proxy,_,_=C.render_with_meta(model,proxy_cam,model.active_sh_degree,absgrad=False)
                    coarse=block_means(fused_loss_map(proxy,proxy_gt),9,15)
                    if mode in ('shared_patch','free_coarse_sort'):
                        loss=patch_weighted_loss(image,target_patches[label],*selected[label])+((1-weights[label])*coarse).mean()
                    else:
                        loss=corrected_loss(blocks,coarse,weights[label])
            loss.backward()
            model.optimizer.step()

        uncached_pair=render_shared(model,cam,masks[0][2],eps2d=C.EPS2D)
        cached_pair=render_shared(model,cam,masks[0][2],eps2d=C.EPS2D,coarse_list=cached16_list)
        assert all(torch.equal(a,b) for a,b in zip(uncached_pair,cached_pair))
        cache_grads=[]
        for fine,proxy in (uncached_pair,cached_pair):
            loss=patch_weighted_loss(fine,target_patches[0],*selected[0])+((1-weights[0])*block_means(fused_loss_map(proxy,proxy_gt),9,15)).mean()
            cache_grads.append(torch.autograd.grad(loss,params))
        cached_gradient_parity={g['name']:discrepancy(a,b) for g,a,b in zip(groups,*cache_grads)}
        for v in cached_gradient_parity.values():
            assert v['relative_l2']<.003 or v['max_abs']<1e-6,v
        del cache_grads,uncached_pair,cached_pair,fine,proxy,loss
        for group in groups: group['lr']=0.
        modes=['native_dense','shared_patch','free_coarse_sort']
        for mode in modes:
            for iteration in range(4):step(mode,iteration)
        timing={mode:[] for mode in modes}
        for repeat in range(3):
            order=modes.copy();random.Random(901+repeat).shuffle(order)
            for mode in order:
                exclusive()
                torch.cuda.synchronize()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                wall=time.perf_counter();start.record()
                for iteration in range(10):step(mode,iteration)
                end.record();torch.cuda.synchronize()
                timing[mode].append({'gpu_ms_per_step':start.elapsed_time(end)/10,'wall_ms_per_step':(time.perf_counter()-wall)*100})
        model.optimizer.zero_grad(set_to_none=True)
        record={'camera_id':camera_id,'shared_fine_native_image_error':image_error,'shared_fine_native_gradient_parity':native_parity,
                'halo_checks':halo_checks,'no_halo_loss_error':missing_halo_error,'mask_build_seconds':mask_seconds,'cached_gradient_parity':cached_gradient_parity, 'list_identity':list_checks, 'intersection_wall_ms':stage_timing,'timings':timing}
        result['records'].append(record);save()
        print('DONE',args.scene,camera_id,{m:round(sum(x['wall_ms_per_step'] for x in ts)/len(ts),3) for m,ts in timing.items()},flush=True)
    result['completed']=True;result['gpu_at_end']=exclusive();save()
    print('COMPLETE',args.scene,args.out,flush=True)


if __name__=='__main__':
    main()

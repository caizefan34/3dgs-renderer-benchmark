"""At one stale state, isolate allocation from native projection VJP precision."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys

import torch

from multilevel_loss import block_means, fused_loss_map
from real_topology_probe import sha
from sort_reuse_renderer import prepare, intersections
from transported_adjoint_predictor import capture_field, transported_nodes
from spatial_moment_predictor import allocate_nodes


def discrepancy(actual, expected):
    diff=actual-expected
    return {'relative_l2':float(diff.norm()/expected.norm().clamp_min(1e-20)),
            'max_abs':float(diff.abs().max()),'difference_l2':float(diff.norm()),
            'reference_l2':float(expected.norm())}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene',required=True)
    parser.add_argument('--gpu',type=int,default=6)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    torch.set_num_threads(4)
    sys.path.insert(0,'/root/epic05_p11')
    import p11_common as C
    gpu=C.assert_clean_gpu(args.gpu)
    uuid=gpu['gpu_info'].split(',')[0].strip()
    torch.cuda.init(); torch.zeros(1,device='cuda'); torch.cuda.synchronize()
    driver_rows=[r for r in C.gpu_compute_apps() if uuid in r]
    assert len(driver_rows)==1
    driver_pid=driver_rows[0].split(',')[0].strip()
    gpu.update(driver_pid=driver_pid,namespace_pid=os.getpid())

    def exclusive():
        rows=[r for r in C.gpu_compute_apps() if uuid in r]
        assert {r.split(',')[0].strip() for r in rows}=={driver_pid},rows

    identity=C.bootstrap_gmain(); C.import_recipe()
    from gaussian_model import GaussianModel
    from gsplat.cuda._wrapper import rasterize_to_pixels
    root=Path(__file__).parent
    checkpoint=Path(C.STATEGEN_DIR)/args.scene/'state_2000.pt'
    previous=json.loads((root/f'{args.scene}_2000_topology.json').read_text())
    assert sha(checkpoint)==previous['checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=GaussianModel(max_sh_degree=3)
    model.restore(saved,C.TRAIN_CFG)
    model.optimizer.load_state_dict(copy.deepcopy(saved['optimizer_state_dict']))
    params=[g['params'][0] for g in model.optimizer.param_groups]
    dataset=C.load_dataset(args.scene)
    camera_id=previous['A'][0]
    sequence=C.camera_sequence_for(args.scene)
    assert sequence is not None
    train_sequence=[int(v) for v in sequence[2000:] if int(v)!=camera_id][:16]
    assert len(train_sequence)==16

    def fine_view(camera_id):
        cam,gt=dataset.get_item(camera_id)
        data=prepare(model,cam,C.EPS2D)
        lists=intersections(data,cam.image_width,cam.image_height)
        latents=tuple(data[k] for k in ('xy','conics','opacity','colors'))
        image,_=rasterize_to_pixels(latents[0],latents[1],latents[3],latents[2],
                                    cam.image_width,cam.image_height,16,lists[3],lists[2],
                                    packed=False,absgrad=False)
        blocks=block_means(fused_loss_map(image[0].clamp(0,1),gt),9,15)
        return cam,data,latents,blocks

    exclusive()
    cam,data,latents,blocks=fine_view(camera_id)
    initial_latent=torch.autograd.grad(blocks.mean(),latents)
    visible=(data['radii'][0]>0).all(-1)
    field=capture_field(data['xy'][0],data['conics'][0],data['opacity'][0],
                        *(g[0] for g in initial_latent),visible)
    del cam,data,latents,blocks,initial_latent

    for step,camera_id_train in enumerate(train_sequence):
        exclusive()
        train_cam,train_gt=dataset.get_item(camera_id_train)
        model.update_learning_rate(2001+step)
        model.optimizer.zero_grad(set_to_none=True)
        _,image,_,_=C.render_with_meta(model,train_cam,model.active_sh_degree,absgrad=False)
        C.fused_total_loss(image,train_gt).backward()
        model.optimizer.step()
        del image
    model.optimizer.zero_grad(set_to_none=True)

    cam,data,latents,blocks=fine_view(camera_id)
    visible=(data['radii'][0]>0).all(-1)
    nodes,color,valid=transported_nodes(field,data['xy'][0],data['conics'][0],data['opacity'][0],visible)
    full_prediction=tuple(sum(node[i] for node in nodes)[None] for i in (1,2,3))+(color[None],)
    labels=(torch.randperm(135,generator=torch.Generator().manual_seed(701)).reshape(9,15)%4).cuda()
    allocated=allocate_nodes(data['xy'][0],nodes,color,valid,labels,cam.image_height,cam.image_width)
    grouped_prediction=tuple(sum(group[i] for group in allocated)[None] for i in range(4))
    latent={name:discrepancy(grouped[0],full[0]) for name,grouped,full in
            zip(('xy','conics','opacity','colors'),grouped_prediction,full_prediction)}

    def vjp(prediction):
        loss=sum((latent*coefficient.detach()).sum() for latent,coefficient in zip(latents,prediction))
        return torch.autograd.grad(loss,params,retain_graph=True)

    reference=vjp(full_prediction)
    repeated=[vjp(full_prediction) for _ in range(3)]
    combined=vjp(grouped_prediction)
    pieces=[vjp(tuple(group[i][None] for group in allocated)) for i in range(4)]
    native_gradient=torch.autograd.grad(blocks.mean(),params)
    groups=[g['name'] for g in model.optimizer.param_groups]
    raw={}
    for index,name in enumerate(groups):
        piece_sum=sum(row[index] for row in pieces)
        raw[name]={'grouped_vs_full':discrepancy(combined[index],reference[index]),
                   'pieces_vs_full':discrepancy(piece_sum,reference[index]),
                   'repeat_vs_full':[discrepancy(row[index],reference[index]) for row in repeated],
                   'full_prediction_vs_native':discrepancy(reference[index],native_gradient[index])}
    result={'scope':'Diagnostic only: same-state latent partition identity and projection VJP replay; no quality/speed claim',
            'scene':args.scene,'camera':camera_id,'cache_age':16,'train_sequence':train_sequence,
            'checkpoint_sha256':sha(checkpoint),'source_sha256':sha(__file__),
            'projection_helper_sha256':sha(root/'sort_reuse_renderer.py'),
            'transport_helper_sha256':sha(root/'transported_adjoint_predictor.py'),
            'allocation_helper_sha256':sha(root/'spatial_moment_predictor.py'),
            'common_sha256':sha(C.__file__),'renderer_identity':identity,'gpu':gpu,
            'latent':latent,'raw':raw,'completed':True,'gpu_at_end':exclusive()}
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE',args.scene,args.out,flush=True)


if __name__=='__main__':
    main()

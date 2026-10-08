import hashlib, json, os, pathlib, subprocess, sys
project=pathlib.Path('/root/codex_sadgs_baseline_20260927')
data=pathlib.Path('/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room')
run=pathlib.Path('/root/sadgs_room_author_5k_l2zero_20260927')
target='/root/miniforge3/envs/sadgs_baseline_clone_20260927/lib/python3.11/site-packages'
source='/root/miniforge3/envs/codex_nerficg/lib/python3.11/site-packages'
sys.path[:0]=[str(project),target,source]
from scene.colmap_loader import read_extrinsics_binary

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
images=sorted(x.name for x in read_extrinsics_binary(data/'sparse/0/images.bin').values())
assert len(images)==311
assert all((data/'images'/n).is_file() for n in images)
train=[n for i,n in enumerate(images) if i%8]
test=[n for i,n in enumerate(images) if i%8==0]
assert (len(train),len(test))==(272,39)
(run/'images.sha256').write_text(''.join(f'{sha(data/"images"/n)}  images/{n}\n' for n in images))
(run/'split.json').write_text(json.dumps({'algorithm':'sorted camera image_name; enumerate; test iff index % 8 == 0','camera_order':images,'train':train,'test':test},indent=2)+'\n')
selected=['README.md','run_train.sh','train.py','render.py','metrics.py','arguments/__init__.py','scene/__init__.py','scene/dataset_readers.py','scene/colmap_loader.py','scene/cameras.py','scene/gaussian_model.py','gaussian_renderer/__init__.py','utils/loss_utils.py','utils/image_utils.py','utils/system_utils.py','submodules/diff-gaussian-rasterization_structgs/setup.py','submodules/simple-knn/setup.py','submodules/fused-ssim/setup.py']
with open(run/'source.sha256','w') as f:
    for name in selected: f.write(f'{sha(project/name)}  {name}\n')
for name in ['images.bin','cameras.bin','points3D.bin','points3D.ply']:
    p=data/'sparse/0'/name
    if p.exists():
        with open(run/'colmap.sha256','a') as f: f.write(f'{sha(p)}  sparse/0/{name}\n')
# Include the repository's tracked tree identity and dirty state.
head=subprocess.check_output(['git','-C',str(project),'rev-parse','HEAD'],text=True).strip()
status=subprocess.check_output(['git','-C',str(project),'status','--porcelain=v1'],text=True)
archive=subprocess.run(['bash','-lc',f"git -C {project} archive --format=tar HEAD | sha256sum"],check=True,text=True,capture_output=True).stdout.split()[0]
# Record Python/Torch and built extension identities without initializing a CUDA context.
import torch
mods=['diff_gaussian_rasterization_structgs._C','simple_knn._C','fused_ssim_cuda']
import importlib
ext={}
for mod in mods:
    m=importlib.import_module(mod)
    path=pathlib.Path(m.__file__)
    ext[mod]={'path':str(path),'sha256':sha(path)}
info={'source_head':head,'source_git_status':status,'source_git_archive_sha256':archive,'python':sys.version,'python_executable':sys.executable,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'extensions':ext,'dataset':str(data),'camera_count':len(images),'train_count':len(train),'test_count':len(test),'split_rule':'lexicographic image_name order; LLFF holdout every 8th camera','image_manifest':'images.sha256','image_manifest_sha256':sha(run/'images.sha256'),'split_manifest':'split.json','split_manifest_sha256':sha(run/'split.json'),'colmap_manifest':'colmap.sha256','source_manifest':'source.sha256'}
(run/'frozen_inputs.json').write_text(json.dumps(info,indent=2)+'\n')

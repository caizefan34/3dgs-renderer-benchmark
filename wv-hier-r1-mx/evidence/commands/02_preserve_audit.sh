set -eu
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
export TMPDIR=/mnt/storage_pool/liaoyuanjun/wv1_feasibility_20261008_2138/tmp
export XDG_CACHE_HOME=/mnt/storage_pool/liaoyuanjun/wv1_feasibility_20261008_2138/cache
export TORCH_EXTENSIONS_DIR=/mnt/storage_pool/liaoyuanjun/wv1_feasibility_20261008_2138/torch_extensions
/home/liaoyuanjun/miniforge3/envs/anysplat/bin/python - <<'PY'
import pathlib,hashlib,json,shutil,torch,os
out=pathlib.Path('/mnt/storage_pool/liaoyuanjun/wv1_feasibility_20261008_2138')
src=pathlib.Path('/dev/shm/wv0_out')
shutil.copytree(src,out/'raw'/'wv0_snapshot')
scripts=out/'raw'/'wv0_live_scripts'; scripts.mkdir()
for p in pathlib.Path('/dev/shm').glob('_wv0*'):
 if p.is_file(): shutil.copy2(p,scripts/p.name)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(8<<20),b''): h.update(block)
 return h.hexdigest()
verified=[]
bundle=src/'WV_HIER_ORACLE0_MX'
for line in (bundle/'MANIFEST.sha256').read_text().splitlines():
 expected,rel=line.split('  ',1); p=bundle/rel
 verified.append({'file':rel,'expected':expected,'actual':sha(p),'pass':sha(p)==expected})
(out/'raw'/'wv0_manifest_check.json').write_text(json.dumps(verified,indent=2))
state_out={}
for p in sorted((src/'states').glob('*.pt')):
 s=torch.load(p,map_location='cpu',weights_only=True)
 state_out[p.stem]=s
 print(p.name,'n_cams',s['n_cams'],'conditions',len(s['conditions']),'sample_metric_type',type(s['conditions']['G16_p0.25']['groups']['means']['nmse']).__name__)
(out/'raw'/'wv0_state_summaries_exact.json').write_text(json.dumps(state_out,indent=2))
pkg=pathlib.Path('/home/liaoyuanjun/.local/lib/python3.10/site-packages/gsplat')
dst=out/'raw'/'gsplat_source_snapshot'
shutil.copytree(pkg,dst,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.so'))
manifest=[]
for p in sorted(dst.rglob('*')):
 if p.is_file(): manifest.append({'file':str(p.relative_to(dst)),'bytes':p.stat().st_size,'sha256':sha(p)})
manifest_text=json.dumps(manifest,sort_keys=True,separators=(',',':'))
(out/'raw'/'gsplat_source_manifest.json').write_text(json.dumps({'tree_sha256':hashlib.sha256(manifest_text.encode()).hexdigest(),'files':manifest},indent=2))
shutil.copy2('/dev/shm/txext/gsplat_cuda/gsplat_cuda.so',out/'raw'/'gsplat_cuda_WV0.so')
shutil.copy2('/dev/shm/txext/gsplat_cuda/.ninja_log',out/'raw'/'wv0_ninja_log.txt')
ckpts=[]
for sid,s in state_out.items():
 p=pathlib.Path(s['ckpt_path'])
 actual=sha(p)
 ckpts.append({'state_id':sid,'path':str(p),'bytes':p.stat().st_size,'sha256':actual,'matches_wv0':actual==s['ckpt_sha256']})
 print('CHECKPOINT',sid,'sha_match',actual==s['ckpt_sha256'])
(out/'raw'/'checkpoint_hash_verification.json').write_text(json.dumps(ckpts,indent=2))
data=pathlib.Path('/mnt/storage_pool/liaoyuanjun/wv0_data')
camera=[]
for sc in ['room','garden','bicycle']:
 p=data/'official'/'mipnerf360'/sc/'cameras.json'
 arr=json.loads(p.read_text())
 target=out/'raw'/'cameras'/sc; target.mkdir(parents=True)
 shutil.copy2(p,target/'cameras.json')
 imgs=[]
 for c in arr[:8]:
  found=[]
  for ext in ['.JPG','.jpg','.png','.jpeg']:
   im=data/'datasets'/'mipnerf360'/sc/'images'/(c['img_name']+ext)
   if im.exists(): found.append({'path':str(im),'sha256':sha(im),'bytes':im.stat().st_size})
  imgs.append({'id':c.get('id'), 'img_name':c.get('img_name'),'images':found})
 camera.append({'scene':sc,'path':str(p),'sha256':sha(p),'n_cameras':len(arr),'selected_first_8':imgs})
(out/'raw'/'camera_image_identity.json').write_text(json.dumps(camera,indent=2))
print('manifest entries',len(verified),'all_match',all(v['pass'] for v in verified))
print('source tree sha',hashlib.sha256(manifest_text.encode()).hexdigest())
print('snapshot preserved',str(out))
PY

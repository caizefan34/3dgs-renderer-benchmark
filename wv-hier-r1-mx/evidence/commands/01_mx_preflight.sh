set -eu
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
OUT=/mnt/storage_pool/liaoyuanjun/wv1_feasibility_20261008_2138
mkdir "$OUT"
mkdir "$OUT/tmp" "$OUT/cache" "$OUT/torch_extensions" "$OUT/raw" "$OUT/logs" "$OUT/commands" "$OUT/gpu_snapshots"
export TMPDIR="$OUT/tmp"
export XDG_CACHE_HOME="$OUT/cache"
export TORCH_EXTENSIONS_DIR="$OUT/torch_extensions"
python3 - <<'PY'
import os, pathlib, json, subprocess, hashlib
out=pathlib.Path('/mnt/storage_pool/liaoyuanjun/wv1_feasibility_20261008_2138')
def call(args):
 p=subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
 return {'argv':args, 'returncode':p.returncode, 'stdout':p.stdout, 'stderr':p.stderr}
cmds=[['hostname'],['date','--iso-8601=seconds'],['nvidia-smi'],['nvidia-smi','--query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.used,utilization.gpu,clocks.current.sm,clocks.current.memory,power.draw','--format=csv'],['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],['df','-B1','/','/mnt/storage_pool'],['df','-i','/','/mnt/storage_pool'],['/usr/bin/nvcc','--version'],['/home/liaoyuanjun/miniforge3/envs/anysplat/bin/nvcc','--version']]
for repo in ['/home/liaoyuanjun/3dgs-renderer-benchmark','/home/liaoyuanjun/.local/lib/python3.10/site-packages/gsplat']:
 for args in [['rev-parse','HEAD'],['branch','--show-current'],['status','--short','--untracked-files=no']]:
  cmds.append(['git','-C',repo]+args)
data={'captures':[call(x) for x in cmds]}
for root in ['/dev/shm/wv0_out','/dev/shm/txext/gsplat_cuda']:
 data[root]=[{'path':str(p),'bytes':p.stat().st_size} for p in sorted(pathlib.Path(root).rglob('*')) if p.is_file()]
for p in [pathlib.Path('/dev/shm/_wv0_campaign_state.py'),pathlib.Path('/dev/shm/_wv0_grad.py'),pathlib.Path('/dev/shm/_wv0_core.py'),pathlib.Path('/dev/shm/txext/gsplat_cuda/gsplat_cuda.so'),pathlib.Path('/dev/shm/txext/gsplat_cuda/build.ninja')]:
 if p.exists():
  data[str(p)]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
  if p.suffix != '.so': (out/'raw'/p.name).write_bytes(p.read_bytes())
(out/'raw'/'mx_preflight.json').write_text(json.dumps(data,indent=2))
print(json.dumps(data,indent=2))
PY
/home/liaoyuanjun/miniforge3/envs/anysplat/bin/python - <<'PY'
import sys,json,torch,importlib.metadata,importlib.util,pathlib
data={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'torch_path':torch.__file__,'torch_cuda':torch.version.cuda,'cxx11_abi':torch._C._GLIBCXX_USE_CXX11_ABI,'torch_config':torch.__config__.show(),'gsplat_version':importlib.metadata.version('gsplat'),'gsplat_spec_origin':importlib.util.find_spec('gsplat').origin}
pathlib.Path('/mnt/storage_pool/liaoyuanjun/wv1_feasibility_20261008_2138/raw/abi.json').write_text(json.dumps(data,indent=2))
print(json.dumps(data,indent=2))
PY

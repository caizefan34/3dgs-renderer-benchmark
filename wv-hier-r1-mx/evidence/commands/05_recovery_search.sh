set -eu
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
OUT=/mnt/storage_pool/liaoyuanjun/wv0_raw_audit_20261008_2148
mkdir "$OUT"
mkdir "$OUT/commands" "$OUT/logs" "$OUT/raw" "$OUT/gpu_snapshots"
find /mnt/storage_pool/liaoyuanjun /home/liaoyuanjun \
  -maxdepth 9 -type f \
  \( -name 'WV0_GRADIENT_RAW.json' \
     -o -name 'WV0_REPORT.md' \
     -o -name 'WV0_VERDICT.json' \
     -o -name 'WV0_GATE_EVALUATION.json' \) \
  -print > "$OUT/raw/requested_find_paths.txt" 2> "$OUT/logs/requested_find_errors.txt" || true
find /mnt/storage_pool/liaoyuanjun /home/liaoyuanjun /dev/shm \
  -maxdepth 9 -type f \
  \( -iname 'WV0_COST_RAW*' -o -iname '*wv0*.tgz' -o -iname '*wv0*.zip' \
     -o -iname '*wv0*.tar*' -o -iname '*wv*hier*.zip' -o -iname '*wv*hier*.tar*' \) \
  -print > "$OUT/raw/supplemental_find_paths.txt" 2> "$OUT/logs/supplemental_find_errors.txt" || true
python3 - <<'PY'
from pathlib import Path
import json,hashlib,tarfile,datetime,re,shutil,subprocess
out=Path('/mnt/storage_pool/liaoyuanjun/wv0_raw_audit_20261008_2148')
prior=Path('/mnt/storage_pool/liaoyuanjun/wv1_feasibility_20261008_2138')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
inventory=[]
paths=set((out/'raw/requested_find_paths.txt').read_text().splitlines())
paths.update((out/'raw/supplemental_find_paths.txt').read_text().splitlines())
paths.update(str(p) for p in Path('/dev/shm/wv0_out').rglob('*') if p.is_file())
for text in sorted(paths):
 p=Path(text)
 if not p.is_file():continue
 item={'path':str(p),'bytes':p.stat().st_size,'mtime_utc':datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat(),'sha256':sha(p),'current_audit_copy':str(prior) in str(p)}
 if p.suffix=='.json':
  try:
   data=json.loads(p.read_text())
   item['json_type']=type(data).__name__;item['schema']=data.get('schema') if isinstance(data,dict) else None
   item['top_level_keys']=list(data) if isinstance(data,dict) else None
   if p.name=='WV0_GRADIENT_RAW.json':
    item['state_ids']=list(data['states']);item['state_condition_count']=sum(len(x['conditions']) for x in data['states'].values())
    item['individual_camera_seed_metric_records']=0;item['stored_seed_metric_arrays']=0
    item['record_layout']='state/condition/group/{rel_l2_per_mask,cos_per_mask,nmse_per_mask}: summary-stat dict'
  except Exception as e:item['parse_error']=str(e)
 inventory.append(item)
archives=[]
for ptext in sorted(paths):
 p=Path(ptext)
 if p.is_file() and (p.name.endswith(('.tgz','.tar.gz','.tar'))):
  entry={'path':str(p),'sha256':sha(p),'members':[]}
  try:
   with tarfile.open(p,'r:*') as t:
    for member in t:
     if not member.isfile():continue
     rec={'name':member.name,'bytes':member.size}
     if 'WV0_GRADIENT_RAW' in member.name:
      b=t.extractfile(member).read();d=json.loads(b)
      rec.update({'sha256':hashlib.sha256(b).hexdigest(),'state_count':len(d['states']),
       'sample_metric_type':type(next(iter(next(iter(d['states'].values()))['conditions'].values()))['groups']['means']['nmse_per_mask']).__name__})
     entry['members'].append(rec)
  except Exception as e:entry['error']=str(e)
  archives.append(entry)
hist=Path('/home/liaoyuanjun/.bash_history')
lines=[]
if hist.exists():
 for i,line in enumerate(hist.read_text(errors='replace').splitlines(),1):
  if re.search(r'\b_wv0|\bwv0_|WV_HIER_ORACLE0|WV0_GRADIENT_RAW',line):
   if re.search(r'password|passwd|token|api.key|secret',line,re.I):line='[omitted: possible credential-bearing command]'
   lines.append({'line':i,'text':line})
(out/'raw/filtered_wv0_shell_history.json').write_text(json.dumps({'scope':'Only WV0-matching commands; all unrelated history omitted','matches':lines},indent=2))
(out/'raw/recovery_file_inventory.json').write_text(json.dumps(inventory,indent=2))
(out/'raw/archive_inventory.json').write_text(json.dumps(archives,indent=2))
# Preserve already collected evidence rather than recomputing experiments.
shutil.copytree(prior/'raw',out/'raw'/'evidence')
for name in ['WV1_WORKCOUNTS.csv','WV1_SOURCE_IDENTITY.json']:
 shutil.copy2(prior/name,out/'raw'/name) if (prior/name).exists() else None
for root in ['/home/liaoyuanjun/3dgs-renderer-benchmark/baseline/reference_v1','/home/liaoyuanjun/3dgs-renderer-benchmark/data/official/mipnerf360']:
 p=Path(root)
 selected=[]
 if p.exists():
  for q in p.glob('*.py'):
   txt=q.read_text(errors='replace')
   if any(x in txt for x in ['llffhold','eval_split','test_indices','cameras.json','train_test_split']):selected.append(q)
  selected += list(p.glob('*/cameras.json'))
 for q in selected:
  dst=out/'raw'/'camera_convention_source'/q.relative_to(Path('/home/liaoyuanjun/3dgs-renderer-benchmark'))
  dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(q,dst)
print(json.dumps({'audit_output':str(out),'inventoried_files':len(inventory),'archives':len(archives),'filtered_history_lines':len(lines),'search_errors_bytes':(out/'logs/requested_find_errors.txt').stat().st_size},indent=2))
PY

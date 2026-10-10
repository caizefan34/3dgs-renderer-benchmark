"""Verify audit deliverables and immutable evidence, then create manifest/ZIP."""
from pathlib import Path
import csv,hashlib,json,datetime,zipfile,sys

root=Path(__file__).resolve().parents[1]
required=['WV0_RAW_INVENTORY.json','WV0_DATA_COMPLETENESS.json','WV0_STATISTICS_AUDIT.md',
 'WV0_ABSOLUTE_COST.csv','WV0_PARETO_RECHECK.csv','WV0_HIERARCHICAL_COST_MODEL.md',
 'WV0_CODEX_READINESS.json','WV0_AUDIT_REPORT.md']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def rows(name):
 with (root/name).open() as f:return list(csv.DictReader(f))
for n in required:assert (root/n).is_file(),n
for n in ['WV0_RAW_INVENTORY.json','WV0_DATA_COMPLETENESS.json','WV0_CODEX_READINESS.json','RUN_STATUS.json']:
 json.loads((root/n).read_text())
c=json.loads((root/'WV0_DATA_COMPLETENESS.json').read_text())
assert c['actual_saved_camera_seed_records']==0 and c['actual_saved_per_seed_metric_arrays']==0
assert c['cartesian_source_implied_count']==9*8*4*2*32==18432
assert len(rows('WV0_ABSOLUTE_COST.csv'))==72 and len(rows('WV0_PARETO_RECHECK.csv'))==72
assert len(rows('raw/GROUP_STATISTICS_AUDIT.csv'))==360
assert all(r['physical_cuda_intersections']=='NOT_MEASURED' for r in rows('WV0_ABSOLUTE_COST.csv'))
assert all(r['forward_backward_ms']=='NOT_MEASURED' for r in rows('WV0_ABSOLUTE_COST.csv'))
assert all(r['raw_export_equals_summary']=='True' for r in rows('raw/GROUP_STATISTICS_AUDIT.csv'))
assert len(rows('raw/evidence/WV1_RECONSTRUCTED_MASKS.csv'))==2304
source=json.loads((root/'raw/evidence/gsplat_source_manifest.json').read_text())
for rec in source['files']:
 p=root/'raw/evidence/gsplat_source_snapshot'/rec['file'];assert sha(p)==rec['sha256'],p
bundle=root/'raw/evidence/wv0_snapshot/WV_HIER_ORACLE0_MX'
for line in (bundle/'MANIFEST.sha256').read_text().splitlines():
 digest,rel=line.split('  ',1);assert sha(bundle/rel)==digest,rel
assert sha(root/'raw/evidence/gsplat_cuda_WV0.so')=='405d4bbdcf168ebedc566b0f064785722cc33c9dd1665c0393880f9ba06538a0'
original=Path('/dev/shm/wv0_out/WV_HIER_ORACLE0_MX')
if original.exists():
 for line in (original/'MANIFEST.sha256').read_text().splitlines():
  digest,rel=line.split('  ',1);assert sha(original/rel)==digest,rel
 assert sha(Path('/dev/shm/txext/gsplat_cuda/gsplat_cuda.so'))=='405d4bbdcf168ebedc566b0f064785722cc33c9dd1665c0393880f9ba06538a0'
checks={'status':'PASS','original_manifest_files':94,'source_snapshot_files':len(source['files']),
 'checkpoint_sha_pass_count':c['checkpoint_sha_pass_count'],'cost_rows':72,'pareto_rows':72,
 'raw_group_summary_rows':360,'mask_rows':2304,'individual_camera_seed_rows':0,
 'timings_fabricated':False,'original_remote_manifest_and_so_reverified':original.exists(),
 'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(root/'raw/AUDIT_VALIDATION.json').write_text(json.dumps(checks,indent=2)+'\n')
files=[]
for p in sorted(root.rglob('*')):
 if p.is_file() and p.name!='SHA256_MANIFEST.json':
  files.append({'file':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
manifest={'schema':'SHA256_MANIFEST.audit.v1','files':files,'self_excluded':True,'archive_external':True}
(root/'SHA256_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
for rec in files:assert sha(root/rec['file'])==rec['sha256']
archive=root.parent/(root.name+'_evidence.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(root.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(root).as_posix())
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print(json.dumps({'validation':checks,'manifest_files':len(files),'archive':str(archive),
 'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive)},indent=2))

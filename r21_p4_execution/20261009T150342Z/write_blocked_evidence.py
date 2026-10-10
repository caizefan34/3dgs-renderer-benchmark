"""Materialize the user-required blocked verdict from recorded observations."""
import csv
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
remote = json.loads((HERE/'MX_PREFLIGHT.json').read_text())
local = json.loads((HERE/'LOCAL_CANDIDATE_VERIFICATION.json').read_text())
assert remote['candidate_availability'] == 'BLOCKED_CANDIDATE_NOT_AVAILABLE'
assert not remote['found_candidate_markers'] and not remote['scan_errors']
manifest = json.loads((HERE/'candidate_identity/R21_CANDIDATE_MANIFEST.json').read_text())
expected = {'gsplat/csrc.so':manifest['frozen_main_binary_sha256'],
            'gsplat/experimental/render/kernels/csrc.so':manifest['validated_R20_extension_sha256'],
            'gsplat/experimental/render/functional/gaussian_inference.py':manifest['R20_wrapper_sha256']}
base_checks = []
for record in remote['paths']:
    for suffix, digest in expected.items():
        if record['path'].endswith('/'+suffix) and record['exists']:
            base_checks.append(dict(path=record['path'],sha256=record['sha256'],expected=digest,
                                    status='PASS_EXISTING_REFERENCE_IDENTITY_ONLY' if record['sha256']==digest else 'FAIL'))
assert len(base_checks)==4 and all(c['status'].startswith('PASS_') for c in base_checks)
phases = [
    ('P4_build','CUDA build / isolated candidate binary identity'),
    ('P4.1','Frozen std_ll vs R21 Radii / Visibility / True AbsGrad / RGB / alpha / raw gradients'),
    ('P4.2','Iteration 1-600 historical statistics and stable-ID split-mask replay'),
    ('P4.3','Bicycle / Room 1000-step smoke'),
    ('P4_performance','Fixed-workload Renderer timing'),
    ('P5','Bicycle / Room paired 5K; output p5_training/ only after all P4 gates pass')]
stages = [dict(stage=key,description=description,status='NOT_EXECUTED',
               reason='P4_NOT_PASSED' if key=='P5' else 'BLOCKED_CANDIDATE_NOT_AVAILABLE')
          for key,description in phases]
status = dict(schema='r21.p4.acceptance-run.v1',status='BLOCKED_CANDIDATE_NOT_AVAILABLE',
              generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              output_root=remote['output_root']['path'],candidate_identity=local,
              mx_repository_head=remote['commands'][1]['stdout'].strip(),
              availability_evidence='preflight/'+HERE.name+'/MX_PREFLIGHT.json',
              stages=stages,existing_reference_identity=base_checks,
              candidate_deployed=False,gpu_kernels_executed=False,training_executed=False,
              training_quality_recovered=False,
              archive_scope='Preflight / blocked evidence only; no P4 or P5 experimental results',
              note='Local delivery identity PASS does not imply delivery to MX or any GPU parity PASS. '
                   'Candidate absence was checked only at the recorded repository and bounded R21 roots. '
                   'Prior R20 / R21 P0-P3 evidence and checkpoints were not modified.')
(HERE/'RUN_STATUS.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
with (HERE/'STAGE_STATUS.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=['stage','description','status','reason']);writer.writeheader();writer.writerows(stages)
with (HERE/'MX_PATH_INSPECTION.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=['path','exists','is_file','is_dir','size','sha256']);writer.writeheader();writer.writerows(remote['paths'])
gpu = next(c for c in remote['commands'] if c['argv'][0]=='nvidia-smi')
(HERE/'MX_GPU_IDENTITY.csv').write_text(gpu['stdout'],encoding='utf-8')
(HERE/'README.md').write_text('''# R21 P4 execution preflight

Status: **BLOCKED_CANDIDATE_NOT_AVAILABLE**.

The local candidate was verified: 31 exact-byte payload hashes and 26 normalized-text
dependency hashes, a clean candidate directory, manifest checksum and Git commits.
The MX repository remains at `32ab80e773f74f4d8e40ff4e338c86b29c7957f5` on `master`.
It does not contain the implementation, validation or delivery commit. No candidate
manifest/build identity was found in the recorded repository and bounded R21 roots.
This is a scoped observation at the timestamp in `MX_PREFLIGHT.json`.

The user explicitly required this blocker when the candidate is not synced to MX.
No candidate deployment, CUDA build, GPU parity, historical replay, training smoke,
performance test or paired 5K was started. All six execution stages are NOT_EXECUTED.
Existing frozen/R20 binary hashes match the pinned identities; this is only a
read-only identity check, not a P4 build or GPU correctness result.

`candidate_identity/` holds audit copies of the local manifest, checksum, patch and
previous local validation report. These copies are evidence, not an installed MX
runtime. Old local PASS entries must not be interpreted as results from this run.

To unblock, sync the sealed delivery commit and its dependencies into a separate
MX checkout, verify its manifest, prepare a new R21 runtime from the validated R20
root, and rerun preflight. Keep this blocked return package immutable; use a new
run subdirectory for subsequent evidence. Never overwrite R20 or R21 P0-P3.

All raw command arguments, stdout/stderr, JSON/CSV inspections and scripts are
included. SHA256SUMS authenticates every evidence file except itself and the
return archive/its checksum; the archive has a separate SHA256 sidecar.
''',encoding='utf-8')
print(json.dumps(dict(status=status['status'],stages=stages),indent=2))

"""Summarize actual evidence and check final-source identity; no GPU claims invented."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
names = ['R20_CODEX_SOURCE_AUDIT.md', 'R20_ABSGRAD_SEMANTICS.md',
         'R20_TRUE_ABSGRAD_DESIGN.md', 'R20_SOURCE_PATCH.diff',
         'R20_GRADIENT_PARITY_TESTS.py', 'R20_ABSGRAD_MICROBENCH.py',
         'R20_R19_TIMING_CODE_AUDIT.md', 'R20_DSH_BUILD_INSTRUCTIONS.md']
assert all((HERE/n).is_file() for n in names)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = json.loads((HERE/'R20_SOURCE_MANIFEST.json').read_text())
for rel, record in manifest['files'].items():
    assert sha(HERE/'source'/rel) == record['patched_sha256']
base = ROOT/'HIGS_SOURCE_HANDOFF_ACTUAL'
export = json.loads((base/'SOURCE_MANIFEST.json').read_text())
assert all(sha(base/r['file']) == r['sha256'] for r in export['files'])
python_files = list(HERE.rglob('*.py'))
for p in python_files: ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
cpu = json.loads((HERE/'R20_TEST_RESULTS.json').read_text())
cuda = json.loads((HERE/'local_checks/compile.json').read_text())
binding = json.loads((HERE/'local_checks/binding_compile.json').read_text())
pixel = json.loads((HERE/'local_checks/blend_check.json').read_text())
kernel = 'gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu'
assert pixel['target_sha256'] == manifest['files'][kernel]['patched_sha256']
assert cpu['status'] == cuda['status'] == binding['status'] == pixel['status'] == 'PASS'
report = dict(status='LOCAL_VALIDATION_PASS_MX_PENDING', date='2026-10-09',
    source_export_files_verified=len(export['files']), frozen_sources_unchanged=True,
    source_overlay_verified=True, deliverables={n: sha(HERE/n) for n in names},
    cpu_checks=cpu, python_ast=dict(status='PASS', files=len(python_files)),
    patch_applicability=dict(status='PASS', command='git apply --check --directory=HIGS_SOURCE_HANDOFF_ACTUAL/source experiments/r20_true_absgrad/R20_SOURCE_PATCH.diff'),
    cuda_translation_unit=cuda, cpp_binding=binding, scoped_pixel_gpu=pixel,
    cli_help=dict(status='PASS', scripts=['R20_ABSGRAD_MICROBENCH.py','R20_PREPARE_MX.py','R20_BUILD_MX.py','R20_MAKE_REAL_FIXTURE.py']),
    frozen_mx_cuda_parity=dict(status='NOT_EXECUTED', reason='Linux frozen binaries/full source unavailable on local Windows host'),
    mx_a100_compile='NOT_EXECUTED', real_scene_parity='NOT_EXECUTED',
    mx_atomic_contention_timing='NOT_EXECUTED', full_training_quality='NOT_EXECUTED',
    r19_timing_root_cause='NOT_ESTABLISHED')
(HERE/'R20_LOCAL_VALIDATION.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({k: report[k] for k in ('status','source_export_files_verified','python_ast','mx_a100_compile','full_training_quality')}, indent=2))

"""Produce a truthful Stage A hard-stop report, with no fabricated CUDA rows."""
from pathlib import Path
import csv
import json
import hashlib
import datetime
import statistics

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/mnt/storage_pool/liaoyuanjun/wv1_feasibility_20261008_2138'
def read(name): return json.loads((ROOT/name).read_text(encoding='utf-8'))
def write(name,data): (ROOT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
counts=read('raw/WV1_COUNT_AUDIT.json')
masks=read('raw/WV1_MASK_AUDIT_SUMMARY.json')
pre=read('raw/mx_preflight.json')
abi=read('raw/abi.json')
source=read('raw/gsplat_source_manifest.json')
ledger=read('raw/WV1_ORIGINAL_COST_LEDGER.json')
with (ROOT/'WV1_WORKCOUNTS.csv').open() as f: work=list(csv.DictReader(f))
with (ROOT/'raw/WV1_GRADIENT_AGGREGATE_RECHECK.csv').open() as f: quality=list(csv.DictReader(f))
with (ROOT/'raw/WV1_BIAS_CORRECTION.csv').open() as f: bias=list(csv.DictReader(f))
so_sha=pre['/dev/shm/txext/gsplat_cuda/gsplat_cuda.so']['sha256']
identity={
 'status':'CURRENT_SOURCE_SNAPSHOT_IDENTIFIED_HISTORICAL_SOURCE_BINARY_BINDING_UNATTESTED',
 'host':'bms-39468022-001','remote_output_directory':REMOTE,
 'observed_repository':{'path':'/home/liaoyuanjun/3dgs-renderer-benchmark','branch':'master',
 'commit':'32ab80e773f74f4d8e40ff4e338c86b29c7957f5',
 'tracked_modified_files':['baseline/reference_v1/colmap_reader.py','baseline/reference_v1/config.py','scripts/phase-r0.1/collect_checkpoints.sh'],
 'note':'Repository commit is NOT the installed gsplat source commit.'},
 'abi':abi,'installed_renderer':{'path':'/home/liaoyuanjun/.local/lib/python3.10/site-packages/gsplat',
 'git_commit':None,'source_manifest':'raw/gsplat_source_manifest.json','source_tree_sha256':source['tree_sha256'],
 'tree_digest_definition':'SHA256 of UTF-8 json.dumps(files,sort_keys=True,separators=(comma,colon)) in sorted path order; excludes .pyc, __pycache__, .so'},
 'observed_existing_binary':{'path':'/dev/shm/txext/gsplat_cuda/gsplat_cuda.so','bytes':23069312,
 'sha256':so_sha,'copied_to':'raw/gsplat_cuda_WV0.so','loaded_or_executed_by_WV1':False,
 'historical_binding':'WV0 records the same path and ABI, but no historical binary/source hashes. The newly observed hash is not retroactive attestation.'},
 'build_ninja_sha256':pre['/dev/shm/txext/gsplat_cuda/build.ninja']['sha256'],
 'new_source_branch':None,'new_worktree':None,'new_cuda_binary':None,
 'reason':'Hard Stage A raw-provenance gate failed before CUDA implementation; source snapshot preserved read-only.',
 'new_gpu_compute_seconds':0,'new_gpu_kernels_launched':0,'new_training_steps':0,
 'checkpoint_verification':'raw/checkpoint_hash_verification.json','camera_identity':'raw/camera_image_identity.json',
 'cache_paths':{k:REMOTE+'/'+v for k,v in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('TORCH_EXTENSIONS_DIR','torch_extensions')]}}
write('WV1_SOURCE_IDENTITY.json',identity)
gates={
 'A1_count':{'status':'CORRECTION_WITH_RAW_RECORD_COUNT_BLOCKED','claimed':9216,'source_and_logs_implied':18432,'saved_individual_camera_seed_records':0,'saved_seed_metric_arrays':0,'saved_state_conditions':72,'saved_metric_stat_objects':1080},
 'A2_loss_IPW':{'status':'PASS_SOURCE_AUDIT_WITH_LABEL_CORRECTION','loss':'RGB squared residual SUM; sparse selected pixels multiplied by 1/p','sampling':'independent Bernoulli per tile, CPU torch.Generator seeds 0..31','MSE_label':'incorrect; fixed division by 3HW absent in both paths','metric_effect':'Common fixed normalization cancels per-camera relative L2, NMSE, cosine; absolute variance rescales.'},
 'A2_bias':{'status':'CORRECTION','report_column':'mean single-mask relative L2, not bias','actual_saved_bias_macro_min':min(float(x['saved_macro_relative_error_of_32_seed_camera_mean']) for x in bias),'actual_saved_bias_macro_max':max(float(x['saved_macro_relative_error_of_32_seed_camera_mean']) for x in bias),'confidence_interval':None,'reason':'Per-seed gradient/metric records absent.'},
 'A2_cosine':{'status':'PASS','note':'cosine similarity rises; 1-cosine difference is negative. No sign reversal.'},
 'A2_variance':{'status':'CORRECTION','note':'E_camera,mask ||g||^2 - ||E_camera,mask g||^2 includes between-camera variation. Not fixed-camera conditional mask variance. Same seed reused across same-size cameras.'},
 'A3_absolute_cost':{'status':'BLOCKED_PHYSICAL_COUNTERS_WITH_PROXY_CORRECTION','counter_scope':'one camera (index 0) per state','relative_G4_G16_proxy_min':min(float(x['relative_to_G16_sparse_proxy']) for x in work if x['G']=='4'),'relative_G4_G16_proxy_max':max(float(x['relative_to_G16_sparse_proxy']) for x in work if x['G']=='4'),'physical_cuda_counts':None,'reason':'No projected tensors or emitted intersection arrays saved; WV0 floor-inclusive boundary proxy differs from CUDA ceil-exclusive rule.'},
 'A4_freeze':{'status':'PARTIAL','checkpoint_sha_matches':9,'original_manifest_matches':94,'unique_cpu_reconstructed_masks':768,'mask_selected_pixel_means_match':72,'missing':['historical camera/seed raw records','historically hashed renderer source/binary identity','historical actual mask hashes','physical CUDA work counters'],'note':'Current camera/image/source identities and regenerated masks saved without relabelling them historical evidence.'},
 'A5_implementation_admission':{'status':'BLOCKED_ORACLE_PROVENANCE','note':'Do not pass a required raw-data/source provenance gate using renamed summary statistics.'}}
write('raw/WV1_STAGE_A_GATES.json',gates)
write('WV1_CORRECTNESS.json',{'status':'NOT_RUN_STAGE_A_BLOCKED','gates':{f'C{i}':{'status':'NOT_RUN','reason':'Stage A prerequisite not passed'} for i in range(1,7)},'B0':{'status':'NOT_VALIDATED','note':'WV0 sparse_tile_grad calls dense render_full and masks loss after rendering; this is not an execution-skipping reference.'},'B1':{'status':'NOT_IMPLEMENTED_STAGE_A_BLOCKED'},'B2':{'status':'NOT_IMPLEMENTED_STAGE_A_BLOCKED'},'ht_source_audit':'raw/WV1_STAGE_A_GATES.json','ht_runtime_gradient_validation':'NOT_RUN','synthetic_only_conclusion':False})
(ROOT/'WV1_PERF_RAW.csv').write_text('arm,state,camera_id,height,width,p,seed,repeat,warmup_count,measurement_index,cuda_fb_ms,step_wall_ms,sampling_ms,projection_ms,isect_ms,sort_ms,raster_forward_ms,backward_ms,loss_ms,peak_vram_bytes,gpu_uuid,contention_status\n',encoding='utf-8')
(ROOT/'WV1_IMPLEMENTATION_DIFF.patch').write_bytes(b'')
write('WV1_PARETO.json',{'status':'NO_PERFORMANCE_PARETO_AVAILABLE','evidence':'DERIVED_FROM_HISTORICAL_SUMMARIES_ONLY','quality_comparisons':quality,'work_counts':'WV1_WORKCOUNTS.csv','G4_relative_NMSE_range':[min(float(x['ratio_G4_G16']) for x in quality),max(float(x['ratio_G4_G16']) for x in quality)],'valid_cross_granularity_W_int_p_claim':False,'F_B_speedup':None,'time_to_quality_speedup':None,'break_even_iteration_ratio':None,'break_even_formula':'steps_B2 / steps_B0 < time_B0 / time_B2','reason':'B0/B1/B2 have no measured matched-source timing in WV1.'})
verdict={'verdict':'BLOCKED_ORACLE_PROVENANCE','stage':'A','performance_GO':False,'algorithmic_DROP':None,'scientific_claim':'Existing aggregates suggest lower frozen-gradient error at matched expected pixels, but do not establish a raw-audited improvement or real time-to-quality benefit.','blockers':['WV0_GRADIENT_RAW.json contains summary-stat dicts, not per-seed arrays; all nine underlying .pt files also discard those arrays.','The 9216 estimate claim has no saved record index supporting it; program and log metadata imply 18432 camera estimates.','Absolute physical isect/sort counts cannot be recovered from structural scalar summaries.','Historical source-to-binary SHA identity was not recorded; current identities are preserved without backdating them.'],'required_to_resume':['Supply historical per-camera/per-seed raw records, physical work evidence and source/binary identity, or explicitly approve a new clearly labelled provenance-repair replay to replace the missing evidence.','Re-evaluate Stage A before creating CUDA arms; continue only after the preflight gate is accepted.'],'microtrajectory':'NOT_AUTHORIZED_BY_GATE','automatic_3K_30K':'NOT_STARTED','existing_environments_overwritten':False,'existing_results_modified':False}
write('WV1_VERDICT.json',verdict)
write('RUN_STATUS.json',{'protocol_id':'WV-HIER-CODEX-FEASIBILITY1-MX','status':'BLOCKED_ORACLE_PROVENANCE','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stages':{'A':'AUDIT_FINISHED_GATE_BLOCKED','B':'NOT_STARTED','C':'NOT_STARTED','D':'NOT_STARTED'},'gpu_compute_seconds_this_run':0,'cuda_kernels_launched_this_run':0,'new_training_steps':0,'performance_measurement_rows':0,'root_available_bytes_at_preflight':0,'pool_available_bytes_at_preflight':32103256064,'remote_output_directory':REMOTE,'local_output_directory':str(ROOT),'implementation_diff_bytes':0,'interpretation':'Empty patch and header-only perf CSV mean NOT IMPLEMENTED / NOT MEASURED, never PASS or zero-time performance.'})
table=[]
for row in work:
 if row['G']=='4':
  s=row['state'];p=float(row['p']);g4=float(row['expected_selected_proxy']);ratio=float(row['relative_to_G16_sparse_proxy'])
  table.append(f'| {s} | {p:.2f} | {g4/ratio:,.2f} | {g4:,.2f} | {ratio:.4f} | {g4*12/1e6:.3f} |')
erratum='''# WV1 — WV0 scientific erratum (original WV0 files preserved)

**CONCLUSION: BLOCKED_ORACLE_PROVENANCE.** This is a hard-preflight finding, not a CUDA speed or correctness result.

## OBSERVED: what was actually retained

All 94 original manifest entries match. All 9 checkpoint SHA256 values match their state summaries. The MX originals are under `/dev/shm/wv0_out/`; complete copies are retained in `raw/wv0_snapshot/`. Original reports, verdict, gate evaluation and actual state `.pt` files were read.

`WV0_GRADIENT_RAW.json` misleadingly describes per-mask metric arrays. Its `nmse_per_mask`, `cos_per_mask` and `rel_l2_per_mask` values are dictionaries with mean/median/std/p25/p75/p90/n. `_wv0_addraw.py` renames the summary dictionaries. The actual `_wv0_campaign_state.py` computes per-mask lists in memory, reduces them with `stats()`, then saves only the summaries.

There are 72 saved state-condition records and 1,080 group×metric statistic objects. **No individual camera/seed metric records and no per-seed metric arrays survive in the audited files.** This does not mean no experiments ran: logs contain successful completion messages for all 9 states. It means the exact observations requested for independent re-aggregation are unavailable.

## CORRECTION: count and aggregation

Source and logs specify 9 states × 8 cameras × 4 granularities × 2 budgets × 32 seeds = **18,432 camera-level estimates**. The per-seed statistic is a mean over 8 cameras; that would be 2,304 camera-averaged seed units if retained. Neither count is 9,216. The observed saved raw record count is zero, and the actual call count cannot be independently recounted from an absent index.

The same seed is reset for every camera. Same-size cameras therefore share the same spatial mask; they are not eight independent mask draws. `std` uses numpy's population convention (ddof=0). The reported variance pools cameras and masks: `E_camera,mask ||g||² − ||E_camera,mask g||²`. It includes between-camera gradient variation and is not the conditional mask variance at a fixed camera.

## CORRECTION: loss, HT, bias and cosine

The implemented dense loss is `sum_RGB,pixel residual²`; sparse loss is `sum mask/p * residual²`. Neither divides by pixels or channels, so “MSE” is a label error. Dividing both by the same fixed `3*H*W` leaves per-camera NMSE, cosine and relative L2 unchanged. Do not divide by the realized selected-pixel count. Border pixels share tile inclusion probability p; Bernoulli tile HT is algebraically valid for this additive loss conditional on fixed pixel gradients. This does not validate the CUDA backward, floating-point equality, or DSSIM.

The report's `Bias_relL2(geom)` column is `geom_rel_l2_mean`: the average single-mask error, not `||mean ghat − dense|| / ||dense||`. The correctly named saved per-group bias is a finite-32-mask error of the camera-mean gradient. Its geometry macro spans **0.0383417–0.2372444** across conditions. It is a unitless relative norm, not a percent or a proof of estimator bias. Raw replicates are absent, so no trustworthy new confidence interval is asserted.

Cosine similarity increases with better alignment. WV0 gate B uses a negative change in `1−cosine`, which is consistent; no sign inversion was found.

## CORRECTION / BLOCKED: absolute work

The saved work values are bounding-box structural counts for **camera 0 only**, not averages over 8 cameras. `_wv0_campaign_state.py` uses floor for the upper bound, clips to `tw−1`, and adds one. CUDA `IntersectTile.cu` uses a ceil exclusive upper bound and clips to `tw`. Example: at G16, mean=(8,8), radii=(8,8), 64×64 image, the old proxy counts 4 tiles whereas the CUDA interval rule gives 1. No projected tensors or actual emitted `isect_ids` were retained to measure the real discrepancy for these states.

The table below multiplies the **saved proxy** by p. It is an expectation under independent tile sampling, not realized work, profiler data, physical `isect_count`, or measured GPU cost. Source shows int64 keys plus int32 values (12 bytes/entry); byte columns exclude input/output duplication, scan buffers, radix-sort scratch and traffic. Actual sparse physical counters remain **NOT MEASURED**.

| state | p | E[G16 proxy] | E[G4 proxy] | G4/G16 | G4 key+value payload MB if proxy held |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(table)+'''

Native G4 expected proxy cost is **4.6572–9.2388×** G16 at equal p, not equal cost. `W_int=p` only normalizes each G to its own dense count. Using G16-dense as denominator, G4 ranges 1.1643–2.3097 at p=.25 and 1.8629–3.6955 at p=.40, so the old `W≤.70` argument does not survive that common denominator. This does not measure hierarchical B2.

For an interior 16×16 parent with 16 independently sampled children, activation is `1−(1−p)^16`: 98.9977404% at .25 and 99.9717889% at .40. Image-border parents have fewer children; reconstructed mask counts include those borders. No 75% parent intersection saving is claimed.

## OBSERVED / DERIVED: reproducibility limits

CPU reconstruction using the saved source, original torch 2.4.1 CPU generator and seeds produced 768 unique masks, expanded into 2,304 state-condition-seed records. All 72 selected-pixel means exactly equal the saved values. Hashes and packed tile bits are provided. These are explicitly **reconstructed now**, not historical mask artifacts.

Recalculation from existing aggregates yields G4/G16 geometry NMSE 0.4149472–0.7071070, with cosine improvement. The normalization label correction does not erase this summary-level signal. Absent raw metric arrays, it cannot be independently re-aggregated or assigned new paired confidence intervals. The absolute work correction invalidates the equal-cost Pareto interpretation.

## FAILURE / CONCLUSION

Stage A cannot PASS on these evidence files. Existing WV0 execution calls `render_full` for every granularity and masks the loss afterward; it is a gradient oracle, not B0/B1/B2 sparse CUDA execution proof. No CUDA arms, timing claims or microtrajectory are admitted. Missing historical evidence must be supplied, or a distinct provenance-repair replay explicitly accepted; regenerated data must never be presented as the original missing records.
'''
(ROOT/'WV1_ORACLE_ERRATUM.md').write_text(erratum,encoding='utf-8')
report=f'''# WV-HIER-CODEX-FEASIBILITY1-MX — Stage A report

**Run status / verdict: BLOCKED_ORACLE_PROVENANCE.** Stage A was audited; its hard gate did not pass. B0 validation, B1 native G4, B2 hierarchical 16→4, C1–C6, F+B timing and microtrajectory are **NOT RUN**. No 3K/30K jobs, source edits, environment installation, or new CUDA compilation occurred.

**CONCLUSION:** The retained summaries suggest improved frozen-gradient statistics for G4 at the same expected pixel fraction. They do not establish a raw-audited result or a plausible measured time-to-quality gain. Native-G4 intersection proxy cost is 4.66–9.24× G16. B2's real work and runtime are unknown. Neither FEASIBILITY_GO nor a definitive algorithmic FEASIBILITY_DROP is supported; the result is BLOCKED.

## OBSERVED: exact locations and identities

- MX host: `bms-39468022-001`, 8×A100-PCIE-40GB, driver 595.71.05.
- Output: `{REMOTE}`.
- Input ZIP SHA256: `b365dabe2c77793dcd11862415dbda1b86ac9727c80bcb99bba0ff102d855f07`; all five listed payload hashes/sizes match.
- Observed project branch: `master`; HEAD `32ab80e773f74f4d8e40ff4e338c86b29c7957f5`. Three existing tracked modifications were preserved. This commit is not the installed gsplat source identity.
- New experiment branch/worktree: **none**, because the prerequisite raw-data gate failed before implementation.
- Current gsplat source-tree SHA256: `{source['tree_sha256']}`. Full per-file hashes and snapshot retained.
- Existing `.so` SHA256: `{so_sha}` (23,069,312 bytes). Copied without execution or overwrite. No new binary exists.
- Runtime: Python 3.10.19, gsplat 1.5.3, torch 2.4.1+cu124, C++11 ABI=0; conda nvcc 12.4.131. System nvcc 11.5 was recorded and not used to build.

The historical WV0 metadata identifies a binary path and ABI but does not contain historical source/binary SHA hashes. Current hashes do not retroactively attest the original run. See [source identity](WV1_SOURCE_IDENTITY.json).

## OBSERVED / DERIVED: principal audit findings

1. Original manifest: **94/94 PASS**. Checkpoints: **9/9 SHA PASS**.
2. Expected camera-level estimate count is **18,432**, not 9,216. There are **0 retained individual camera/seed records**, 72 state-condition summaries, and 1,080 group-metric summary-stat objects.
3. The file called “RAW” simply renames aggregate dictionaries. Underlying `.pt` files also lack the arrays. This prevents independent reconstruction of paired gradient statistics and intervals.
4. Loss is RGB squared-error **sum**, not MSE. Source-level Bernoulli IPW is correct for this additive loss. Fixed common pixel/channel normalization preserves relative metrics.
5. “Bias” report column contains average single-mask relative error. Saved finite-sample mean-gradient error is a different quantity. Variance pools both camera and mask variation.
6. Cost counts cover camera 0 only and use a boundary rule different from CUDA. The corrected absolute proxy table is retained, but physical intersections/sort bytes are **unknown**.
7. **768 unique masks** were reconstructed on CPU, covering all 2,304 state-condition-seed combinations; selected pixel means agree **72/72**. Hashes and packed bits explicitly carry reconstruction provenance.
8. G4/G16 aggregate geometry NMSE is **0.414947–0.707107**; the absolute intersection proxy ratio is **4.657154–9.238840**. Better error per supervised pixel is not evidence of lower elapsed training cost.

Detailed formulas, all 18 G16/G4 state-budget comparisons, corrections and scope limits: [WV1_ORACLE_ERRATUM.md](WV1_ORACLE_ERRATUM.md).

## OBSERVED: resource cost and protection

At preflight, root had **0 available bytes**; pool had 32,103,256,064 bytes (29.90 GiB). All new remote outputs and cache directories use the authorized pool. GPUs 0–3 were occupied; 4–7 were idle at the snapshots. No processes were stopped and no GPU was allocated for WV1.

**WV1 new GPU compute: 0 seconds; new CUDA kernels: 0; training steps: 0.** All work was file inspection, hashing, and CPU mask/statistic reconstruction. No dollar cost is estimated without a rate. Original WV0 launcher logs sum to **25,685.7 single-GPU process-wall seconds (7.1349 hours)**; this historical figure includes CPU/I/O and is not a GPU-active-time or bill measurement. Full records: `raw/WV1_ORIGINAL_COST_LEDGER.json`.

## FAILURE: why later stages were not executed

The user-requested execution prompt makes WV0 raw provenance and exact source identity hard preconditions. File names alone do not satisfy the required per-mask raw data, physical-work evidence, or historical source/binary attestation. These evidence gaps cannot be filled by treating means/quantiles as observations or by relabelling a new run as WV0. The supplied request authorizes a gated feasibility test, not bypassing the failed gate.

This is an **evidence blocker**, not evidence that B2 is slow or incorrect. No tests were marked PASS without execution. `WV1_IMPLEMENTATION_DIFF.patch` is empty; `WV1_PERF_RAW.csv` contains a header and **zero measurements**. `WV1_CORRECTNESS.json` explicitly marks C1–C6 NOT_RUN. Occupancy/register/warp-idle statistics, peak VRAM and memory-safety validation are likewise NOT_RUN.

## HYPOTHESIS / next admissible step

B2 might reduce fine-child blend/backward work while keeping nearly all parent bin/sort work. That remains a hypothesis. Supply the missing original records and historical identities, or explicitly accept a new separately labelled provenance-repair replay. A new replay must save camera/seed metrics, exact input/mask hashes, real intersection/sort counts and source/binary identity before reevaluating the Stage A gate. No GPU replay or training campaign is automatically scheduled here.

If later B2 is slower per step, the necessary break-even condition is `steps_B2/steps_B0 < time_B0/time_B2`. No numerical threshold can be reported without measured paired times.

## Evidence index

- `raw/wv0_snapshot/`: immutable copy of all located WV0 outputs, including original state `.pt` and detailed execution logs.
- `raw/wv0_live_scripts/`: runner, loader, estimator, aggregation and report scripts read on MX.
- `raw/gsplat_source_snapshot/`, `raw/gsplat_source_manifest.json`, `raw/gsplat_cuda_WV0.so`, `raw/build.ninja`: current source and binary evidence.
- `raw/checkpoint_hash_verification.json`, `raw/camera_image_identity.json`, `raw/cameras/`: input identities.
- `raw/WV1_COUNT_AUDIT.json`, `raw/WV1_STAGE_A_GATES.json`, corrected bias/quality CSVs: audit calculations.
- `raw/reconstructed_masks/`, `raw/WV1_RECONSTRUCTED_MASKS.csv`: regenerated mask bits and hashes.
- `WV1_WORKCOUNTS.csv`: all 72 state/G/p proxy comparisons with explicit NOT_MEASURED physical fields.
- `commands/`, `logs/`, `gpu_snapshots/`: exact audit commands, complete captured output, GPU/device/storage evidence.
- `SHA256_MANIFEST.json`: delivered-file hashes; archive hash recorded separately to avoid self-reference.

All available evidence is delivered; the explicitly missing historical records are not claimed to be included.
'''
(ROOT/'WV1_REPORT.md').write_text(report,encoding='utf-8')
for capture in pre['captures']:
 if capture['argv'][0]=='nvidia-smi':
  suffix='full' if len(capture['argv'])==1 else ('processes' if 'compute-apps' in capture['argv'][1] else 'devices')
  (ROOT/'gpu_snapshots'/f'preflight_{suffix}.txt').write_text(capture['stdout'],encoding='utf-8')
print('Stage A report and required non-fabricated deliverables written.')

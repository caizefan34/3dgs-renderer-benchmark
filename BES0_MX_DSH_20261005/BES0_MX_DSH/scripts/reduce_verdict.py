#!/usr/bin/env python3
"""Read-only evidence reduction followed by one local VERDICT.json write.
No scientific imports or execution. Templates/unknowns fail closed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from provenance import ROOT,protocol_digest,implementation_digest,reference_digest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('review_dir',type=Path)
    a=p.parse_args(); review=a.review_dir.resolve()
    if not review.is_dir():
        p.error('review directory must exist')
    issues=[]; stages={}; reasons=[]
    expected_protocol=protocol_digest(); expected_impl=implementation_digest()
    try:
        selected_path=review/'selected_config.json'
        selected=json.loads(selected_path.read_text())
        selected_hash=sha(selected_path)
        if selected.get('status')!='FROZEN':
            issues.append('selected config is not FROZEN')
        if selected.get('protocol_sha256')!=expected_protocol or selected.get('implementation_sha256')!=expected_impl:
            issues.append('selected config fingerprint differs')
        if selected.get('gamma')!=2 or selected.get('tile') not in (8,16) or selected.get('train_samples') not in (1,2,4):
            issues.append('selected config outside frozen search')
        if selected.get('bound_mode') not in ('R0_ENUMERATED_NATIVE','R1_CERTIFIED_INTERVAL'):
            issues.append('unknown bound mode')
        if selected.get('bound_mode')=='R1_CERTIFIED_INTERVAL' and not selected.get('bound_certificate_id'):
            issues.append('R1 certificate id missing')
    except (OSError,ValueError,TypeError):
        issues.append('selected config missing/invalid'); selected_hash=None
    for name in ('E0','E1','E2','E3'):
        try:
            obj=json.loads((review/(name+'.json')).read_text())
        except (OSError,ValueError):
            issues.append(name+' missing/invalid'); continue
        stages[name]=obj
        if obj.get('schema')!='bes0.stage.v1' or obj.get('stage')!=name:
            issues.append(name+' schema/stage differs')
        if obj.get('protocol_sha256')!=expected_protocol:
            issues.append(name+' protocol differs')
        if name=='E0':
            if obj.get('reference_sha256')!=reference_digest():
                issues.append('E0 reference differs')
        else:
            if obj.get('implementation_sha256')!=expected_impl:
                issues.append(name+' implementation differs (rerun production stages after edits)')
            if obj.get('selected_config_sha256')!=selected_hash:
                issues.append(name+' selected config differs')
            if not isinstance(obj.get('source_commit'),str) or not obj['source_commit']:
                issues.append(name+' source commit missing')
        raw=obj.get('raw_files')
        if not isinstance(raw,list) or not raw:
            issues.append(name+' raw evidence missing'); continue
        kinds=set()
        for item in raw:
            try:
                raw_path=review/item['path']
                resolved=raw_path.resolve()
                if raw_path.is_symlink() or not resolved.is_relative_to(review) or not resolved.is_file():
                    raise ValueError('unsafe/missing')
                if sha(resolved)!=item['sha256']:
                    raise ValueError('hash mismatch')
                kinds.add(item.get('kind'))
                if item.get('kind')=='guard':
                    guard=json.loads(resolved.read_text())
                    target=json.loads((ROOT/'config/mx_target.json').read_text())
                    if guard.get('hostname','').split('.')[0]!=target['expected_hostname']:
                        issues.append(name+' raw guard host differs')
                    if guard.get('target_config_sha256')!=sha(ROOT/'config/mx_target.json'):
                        issues.append(name+' raw guard target differs')
                    if sha(resolved)!=obj.get('guard_sha256'):
                        issues.append(name+' guard hash differs')
                if item.get('kind')=='run_status':
                    status=json.loads(resolved.read_text())
                    if status.get('exit_code')!=0 and obj.get('status')=='PASS':
                        issues.append(name+' PASS despite failed launcher')
            except (KeyError,OSError,ValueError,TypeError):
                issues.append(name+' raw evidence unsafe/missing/hash differs')
        if not {'guard','run_status'}.issubset(kinds):
            issues.append(name+' MX launcher evidence missing')
        if name!='E0' and not {'guard','run_status','measurements','source_patch'}.issubset(kinds):
            issues.append(name+' guard/status/measurements/source_patch evidence incomplete')
    # Trust neither a claimed PASS nor unpopulated templates.
    required={
      'E0':['color_expectation','gradient_expectation','analytic_bounds_reference',
            'first_hit_distribution','candidate_count_identity','candidate_count_bound',
            'trial_tail_bound','suffix_distribution','wrong_hazard_ratio_rejected',
            'empty_random_set','certain_background'],
      'E1':['native_semantics_frozen','native_wrapper_correct','bounds_certified',
            'routing_partition_correct','prefix_numerics_correct','rng_streams_independent',
            'native_small_fixture_gradient_correct','environment_verified'],
      'E2':['full_parameter_backward_correct','matched_optimizer_and_data',
            'complete_step_timing','strong_baselines_present','gradient_risk_measured'],
      'E3':['configuration_frozen','paired_inputs','fixed_n_only','quality_metrics_frozen',
            'cold_and_warm_times_reported','no_test_tuning']}
    correctness_failed=[]
    for name,obj in stages.items():
        checks=obj.get('checks',{})
        if not isinstance(checks,dict):
            issues.append(name+' checks missing'); continue
        for key in required[name]:
            v=checks.get(key)
            if v is False:
                correctness_failed.append(name+':'+key)
            elif v is not True:
                issues.append(name+':'+key+' unknown')
        if obj.get('inputs_verified') is not True:
            issues.append(name+' inputs not verified')
    def neednum(obj,key,stage):
        value=obj.get(key)
        if not finite_number(value):
            issues.append(stage+':'+key+' missing'); return None
        return value
    if 'E0' in stages:
        for key,limit in [('color_maxabs',1e-10),('gradient_maxabs',1e-9)]:
            v=neednum(stages['E0'],key,'E0')
            if v is not None and v>limit:
                correctness_failed.append('E0:'+key)
    e1=stages.get('E1',{}); e2=stages.get('E2',{}); e3=stages.get('E3',{})
    bound=neednum(e1,'bound_violation_count','E1')
    wrapper=neednum(e1,'native_wrapper_color_maxabs','E1')
    if bound is not None and bound!=0:
        correctness_failed.append('E1:bounds')
    if wrapper is not None and wrapper>1e-5:
        correctness_failed.append('E1:wrapper')
    opportunities=e1.get('scenes',[]); steps=e2.get('scenes',[]); pairs=e3.get('pairs',[])
    if len(opportunities)!=3 or len(steps)!=3 or len(pairs)!=9:
        issues.append('must have 3 E1 scenes, 3 E2 scenes, 9 E3 pairs')
    scene_ids={x.get('scene_id') for x in opportunities if isinstance(x,dict)}
    if len(scene_ids)!=3 or None in scene_ids:
        issues.append('three unique scene ids required')
    if {x.get('scene_id') for x in steps if isinstance(x,dict)}!=scene_ids:
        issues.append('E2 scenes differ')
    opp=[]; step_ratios=[]; memory_ratios=[]; pair_ratios=[]
    for item in opportunities:
        val=neednum(item,'optimistic_speedup','E1')
        if val is not None: opp.append(val)
    for item in steps:
        b=neednum(item,'baseline_step_ms','E2'); c=neednum(item,'candidate_step_ms','E2')
        bm=neednum(item,'baseline_peak_process_mib','E2'); cm=neednum(item,'candidate_peak_process_mib','E2')
        if b is not None and c is not None and b>0 and c>0: step_ratios.append(b/c)
        else: issues.append('E2 positive times missing')
        if bm is not None and cm is not None and bm>0 and cm>0: memory_ratios.append(cm/bm)
        else: issues.append('E2 positive memory missing')
        groups=item.get('timing_groups')
        if not isinstance(groups,list) or len(groups)!=5:
            issues.append('E2 5 timing groups missing')
    identities=[]
    quality_limits={'mean_psnr_loss_db':.2,'mean_ssim_loss':.005,'mean_lpips_increase':.01,
                    'worst_view_psnr_loss_db':1.,'worst_view_lpips_increase':.03}
    quality_fail=[]
    for item in pairs:
        identities.append((item.get('scene_id'),item.get('seed')))
        b=neednum(item,'baseline_time_to_quality_s','E3'); c=neednum(item,'candidate_time_to_quality_s','E3')
        if item.get('status')!='REACHED':
            quality_fail.append(str(identities[-1])+':target not reached')
        if item.get('initial_checkpoint_reaches_target') is not False:
            issues.append('E3 target indistinguishable/unknown at initial checkpoint')
        if b is not None and c is not None and b>0 and c>0: pair_ratios.append(b/c)
        elif item.get('status')=='REACHED': issues.append('E3 positive times missing')
        for key,limit in quality_limits.items():
            v=neednum(item,key,'E3')
            if v is not None and v>limit: quality_fail.append(str(identities[-1])+':'+key)
        bm=neednum(item,'baseline_peak_process_mib','E3'); cm=neednum(item,'candidate_peak_process_mib','E3')
        if bm is not None and cm is not None and bm>0 and cm>0: memory_ratios.append(cm/bm)
        else: issues.append('E3 positive memory missing')
    if set(identities)!={(s,k) for s in scene_ids for k in (17,29,43)} or len(set(identities))!=9:
        issues.append('E3 pair identities incomplete/duplicated')
    geomean=math.exp(sum(math.log(x) for x in pair_ratios)/len(pair_ratios)) if pair_ratios else None
    statuses=[x.get('status') for x in stages.values()]
    explicit_blocks=[s for s in statuses if isinstance(s,str) and (s.startswith('BLOCKED_') or s.startswith('INCONCLUSIVE_'))]
    if correctness_failed:
        verdict='NO_GO_CORRECTNESS'; reasons=correctness_failed
    elif explicit_blocks:
        verdict=explicit_blocks[0]; reasons=explicit_blocks+issues
    elif any(stages.get(s, {}).get("cuda_implemented") is False for s in ("E1", "E2", "E3")):
        verdict="INCONCLUSIVE_CUDA_NOT_IMPLEMENTED"
        reasons=["production CUDA renderer not implemented (cuda_implemented=false); "
                 "Torch-reference timings are NOT CUDA evidence and cannot support a GO/NO_GO"] + issues
    elif issues:
        verdict='INCONCLUSIVE_EVIDENCE'; reasons=issues
    elif all(x<3 for x in opp) or any(x<3 for x in step_ratios):
        verdict='NO_GO_OPPORTUNITY'; reasons=['optimistic or actual full-step investment gate not met']
    elif quality_fail or any(x>1 for x in memory_ratios) or any(x<2 for x in pair_ratios) or geomean<3:
        verdict='NO_GO_FIXED_N_PILOT'; reasons=quality_fail+['quality/memory/time-to-quality gate not met']
    elif statuses!=['PASS']*4:
        verdict='INCONCLUSIVE_EVIDENCE'; reasons=['stage status not PASS']
    else:
        verdict='GO_FIXED_N_PILOT'; reasons=['fixed-N pilot only; full densification not validated']
    result={'schema':'bes0.verdict.v1','verdict':verdict,'reasons':reasons,
            'protocol_sha256':expected_protocol,'implementation_sha256':expected_impl,
            'selected_config_sha256':selected_hash,'e3_geomean_speedup':geomean,
            'e3_pair_speedups':pair_ratios,'e2_step_speedups':step_ratios,
            'render_same_quality_success':e3.get('render_same_quality_success','NOT_RUN'),
            'limitations':['reducer verifies evidence shape/hashes, not measurement honesty',
                           'candidate novelty and disruptive impact unproven']}
    (review/'VERDICT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return 0 if verdict=='GO_FIXED_N_PILOT' else 2


if __name__=='__main__':
    raise SystemExit(main())

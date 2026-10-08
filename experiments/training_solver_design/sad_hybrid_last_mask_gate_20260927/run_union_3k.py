#!/usr/bin/env python3
import datetime, hashlib, json, os, pathlib, re, signal, subprocess, sys, threading, time
PROJECT=pathlib.Path('/root/codex_sadgs_baseline_20260927')
DATA=pathlib.Path('/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room')
RUN=pathlib.Path('/root/sadgs_room_unionmask_3k_l2zero_20260928')
OUT=RUN/'model'
LOGS=RUN/'logs'
GPU_INDEX='3'
GPU_UUID='GPU-431ac351-20f8-bdf7-d798-1c9a3f545b72'
GPU_LIMIT_MIB=73728
STAGE_TIMEOUT_S=21600
SAMPLE_S=1.0
TARGET='/root/miniforge3/envs/sadgs_baseline_clone_20260927/lib/python3.11/site-packages'
SOURCE='/root/miniforge3/envs/codex_nerficg/lib/python3.11/site-packages'
PY='/root/miniforge3/envs/sadgs_baseline_clone_20260927/bin/python'

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def runout(cmd): return subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT).strip()
def gpu_stats():
    line=runout(['nvidia-smi','-i',GPU_INDEX,'--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
    uuid,used,util=[x.strip() for x in line.split(',')]
    if uuid!=GPU_UUID: raise RuntimeError(f'GPU UUID changed: {uuid}')
    return int(used),int(util)
def gpu_pids():
    raw=runout(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'])
    ret=[]
    for line in raw.splitlines():
        parts=[x.strip() for x in line.split(',')]
        if len(parts)>=3 and parts[0]==GPU_UUID:
            try: ret.append({'pid':int(parts[1]),'used_mib':int(parts[2])})
            except ValueError: pass
    return ret
def git_identity():
    head=runout(['git','-C',str(PROJECT),'rev-parse','HEAD'])
    status=runout(['git','-C',str(PROJECT),'status','--porcelain=v1'])
    archive=subprocess.run(['bash','-lc',f'git -C {PROJECT} archive --format=tar HEAD | sha256sum'],check=True,text=True,capture_output=True).stdout.split()[0]
    return {'head':head,'status':status,'archive_sha256':archive}
def check_images():
    for line in (RUN/'images.sha256').read_text().splitlines():
        expected,rel=line.split('  ',1)
        if sha(DATA/rel)!=expected: return False,rel
    return True,None

def main():
    RUN.mkdir(exist_ok=True); LOGS.mkdir(exist_ok=True)
    frozen=json.loads((RUN/'frozen_inputs.json').read_text())
    if frozen['source_head']!='029814fb5ed773e4a794cec5115951a64f610ab8': raise RuntimeError('unexpected source HEAD')
    if frozen['source_git_status']!='': raise RuntimeError('source checkout not clean before run')
    if sha(RUN/'images.sha256')!=frozen['image_manifest_sha256']: raise RuntimeError('image manifest hash mismatch')
    image_ok,bad=check_images()
    if not image_ok: raise RuntimeError(f'dataset image changed before run: {bad}')
    if OUT.exists(): raise RuntimeError(f'output already exists: {OUT}')
    used,util=gpu_stats()
    pids=gpu_pids()
    if used>256 or util>10 or pids: raise RuntimeError(f'GPU preflight failed: used={used} MiB util={util}% pids={pids}')
    env=os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES':GPU_INDEX,'CUDA_HOME':'/usr/local/cuda-12.8','TORCH_CUDA_ARCH_LIST':'8.0','PYTHONPATH':f'{PROJECT}:{TARGET}:{SOURCE}','PYTHONDONTWRITEBYTECODE':'1'})
    train=[PY,'/root/sad_hybrid_unionmask_trial_20260928/train_union_mask.py','-s',str(DATA),'-i','images','--eval','--scale_rotation_scheduler','--adam_eps_order','10','--freq_transmittance_threshold','0.8','--freq_opacity_threshold','0.5','--batch_size','2','--optimizer_type','hybrid','--sample_bbox_faces','--warmup_densification','--mult','0.25','--iterations','3000','--position_lr_max_steps','3000','--opacity_reset_interval','1200','--densification_interval','500','--densify_until_iter','1900','--densify_from_iter','100','--save_iterations','3000','--prune_iterations','1900','--lambda_l2','0.0','--model_path',str(OUT)]
    render=[PY,'render.py','-m',str(OUT),'--skip_train','--iteration','3000']
    metrics=[PY,'metrics.py','-m',str(OUT)]
    stages=[('train',train,{**env,'OAR_JOB_ID':'room'}),('render',render,env),('metrics',metrics,env)]
    OUT.mkdir()
    meta={'status':'running','started_utc':utc(),'host':runout(['hostname']),'project_root':str(PROJECT),'dataset':str(DATA),'output':str(OUT),'source_before':git_identity(),'gpu':{'index':int(GPU_INDEX),'uuid':GPU_UUID,'name':'NVIDIA A100-SXM4-80GB','total_mib':81920,'preflight_used_mib':used,'preflight_util_percent':util,'preflight_compute_pids':pids,'memory_gate_mib':GPU_LIMIT_MIB,'sample_interval_seconds':SAMPLE_S},'allocator_cap':'none','stage_timeout_seconds':STAGE_TIMEOUT_S,'commands':{n:c for n,c,_ in stages},'environment':{'CUDA_VISIBLE_DEVICES':GPU_INDEX,'PYTHONPATH':f'{PROJECT}:{TARGET}:{SOURCE}','PYTHONDONTWRITEBYTECODE':'1','CUDA_HOME':env['CUDA_HOME'],'TORCH_CUDA_ARCH_LIST':env['TORCH_CUDA_ARCH_LIST']},'phases':[],'peak_gpu_used_mib':used,'peak_host_process_rss_kib':0,'gpu_samples_file':'gpu_samples.csv'}
    (RUN/'gpu_samples.csv').write_text('utc,phase,used_mib,util_percent,pids\n')
    def sample(phase,pid):
        used,util=gpu_stats(); pids=gpu_pids()
        meta['peak_gpu_used_mib']=max(meta['peak_gpu_used_mib'],used)
        # Do not attribute nvidia-smi host PIDs to container PIDs: EPIC-05 namespaces differ.
        try:
            stat=pathlib.Path(f'/proc/{pid}/status').read_text()
            m=re.search(r'^VmHWM:\s+(\d+)\s+kB',stat,re.M)
            if m: meta['peak_host_process_rss_kib']=max(meta['peak_host_process_rss_kib'],int(m.group(1)))
        except FileNotFoundError: pass
        with open(RUN/'gpu_samples.csv','a') as f: f.write(f'{utc()},{phase},{used},{util},{json.dumps(pids,separators=(",",":"))}\n')
        return used
    t0=time.monotonic()
    failure=None
    for name,cmd,stage_env in stages:
        phase={'name':name,'command':cmd,'started_utc':utc(),'status':'running'}
        log_path=LOGS/f'{name}.log'
        with open(log_path,'wb') as out:
            p=subprocess.Popen(cmd,cwd=PROJECT,env=stage_env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            phase['pid']=p.pid
            phase_start=time.monotonic()
            while p.poll() is None:
                try:
                    used_now=sample(name,p.pid)
                    if used_now>=GPU_LIMIT_MIB: failure=f'gpu_memory_gate:{used_now}MiB >= {GPU_LIMIT_MIB}MiB'
                    if time.monotonic()-phase_start>STAGE_TIMEOUT_S: failure=f'stage_timeout:{name}:{STAGE_TIMEOUT_S}s'
                except Exception as e: failure=f'monitor_error:{type(e).__name__}:{e}'
                if failure:
                    try: os.killpg(p.pid,signal.SIGTERM)
                    except ProcessLookupError: pass
                    try: p.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        try: os.killpg(p.pid,signal.SIGKILL)
                        except ProcessLookupError: pass
                    break
                time.sleep(SAMPLE_S)
            rc=p.wait()
        phase.update({'ended_utc':utc(),'wall_seconds':round(time.monotonic()-phase_start,3),'return_code':rc,'log':str(log_path),'status':'passed' if rc==0 and not failure else 'failed'})
        meta['phases'].append(phase)
        if rc!=0 or failure:
            if not failure: failure=f'{name}_exit_code:{rc}'
            break
    meta['ended_utc']=utc()
    meta['total_process_wall_seconds']=round(time.monotonic()-t0,3)
    meta['failure_reason']=failure
    meta['source_after']=git_identity()
    meta['dataset_images_unchanged_after']=check_images()
    meta['status']='passed' if not failure and len(meta['phases'])==3 else 'failed'
    if meta['status']=='passed':
        test_names=set(json.loads((RUN/'split.json').read_text())['test'])
        expected_names={f'{i:05d}.png' for i in range(len(test_names))}
        render_dir=OUT/'test'/'ours_3000'/'renders'
        gt_dir=OUT/'test'/'ours_3000'/'gt'
        renders={p.name for p in render_dir.iterdir() if p.is_file()}
        gts={p.name for p in gt_dir.iterdir() if p.is_file()}
        if renders!=expected_names or gts!=expected_names: failure=f'author numeric render/gt names or count mismatch: renders={len(renders)} gt={len(gts)} expected={len(expected_names)}'
        else:
            results=json.loads((OUT/'results.json').read_text())
            per_view=json.loads((OUT/'per_view.json').read_text())
            methods=results
            if 'ours_3000' not in methods: failure='results missing ours_3000'
            elif not {'PSNR','SSIM','LPIPS'}<=set(methods['ours_3000']): failure='results missing one or more PSNR/SSIM/LPIPS metrics'
            elif any(len(per_view['ours_3000'][metric])!=39 for metric in ('PSNR','SSIM','LPIPS')): failure='per-view metrics do not cover all 39 held-out images'
            else: meta['quality_metrics']=methods['ours_3000']
        if not failure:
            ply=OUT/'point_cloud/iteration_3000/point_cloud.ply'
            meta['final_ply']={'path':str(ply),'size_bytes':ply.stat().st_size,'sha256':sha(ply)}
            txt=(LOGS/'train.log').read_text(errors='replace')
            match=re.search(r'Training time:\s*([0-9.]+)',txt)
            meta['author_cuda_event_training_seconds']=float(match.group(1)) if match else None
            meta['rendered_test_views']=len(renders)
            meta['held_out_names_sha256']=sha(RUN/'split.json')
    if failure: meta['status']='failed'; meta['failure_reason']=failure
    if meta['source_after']!=meta['source_before']: meta['source_changed_during_run']=True
    with open(RUN/'run.meta.json','w') as f: json.dump(meta,f,indent=2)
    print(json.dumps(meta,indent=2),flush=True)
    return 0 if meta['status']=='passed' else 1

if __name__=='__main__':
    try: sys.exit(main())
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        out={'status':'preflight_or_supervisor_error','ended_utc':utc(),'error':f'{type(e).__name__}: {e}'}
        pathlib.Path('/root/sadgs_room_unionmask_3k_l2zero_20260928/run.meta.json').write_text(json.dumps(out,indent=2)+'\n')
        print(json.dumps(out,indent=2),file=sys.stderr,flush=True)
        sys.exit(2)

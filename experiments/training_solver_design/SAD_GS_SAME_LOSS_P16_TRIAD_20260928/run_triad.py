#!/usr/bin/env python3
"""Run the frozen three-scene SAD-GS original-loss calibration on EPIC-05 GPU3."""
import csv, datetime, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path('/root/sadgs_same_loss_p16_20260928')
REPO = Path('/root/codex_sadgs_baseline_20260927')
DATA = Path('/root/epic05_p16rb/common_data/mipnerf360')
ART = Path('/root/epic05_p16rb/artifacts')
P16 = Path('/root/epic05_p16rb')
PY = '/root/miniforge3/envs/sadgs_baseline_clone_20260927/bin/python'
METPY = '/root/miniforge3/envs/gsplat/bin/python'
EXTS = {
 'diff_gaussian_rasterization_structgs._C': '2e6949550515397332369cc79791b85b3760bd8b7cf2b4b6b6888c991682b79b',
 'simple_knn._C': '400577975aab0b0e183488c9fc710dbbbf9b33255116090c3210a6aca8795e72',
 'fused_ssim_cuda': 'd3601b94bc216c43c91e62c7094583ccf46b67a8a37ab7e66d00b85249ff7c6f',
}
EXPECTED_DATA = {
 'room': ('3616f0ba74a1d14600c148c5e9c9025e2638427a3214c80000f147bf053475b8', 315),
 'bicycle': ('add28de326c4e246d882f5395345ac62510f948748963c569fa0c0dba6653d1c', 198),
 'garden': ('ec8240060fcce5b17eb6915b3bedbde08d6ebce3fc8a64338d731bf60650949c', 189),
}
EXPECTED_SPLITS = {
 'room': '971e5f9d5075520ea7290e9efe52fb9d8b08b7fad4633b14a0889a813c51591a',
 'bicycle': '31d12396f6336b03b6f458f3b215cae6c6330e137a77aed76eaf3b2557b36914',
 'garden': '2d70a3baa5b5613cb0ed3288a869b22b0b57398cba78b5416d93da1a8e7a2e07',
}
FLAGS = {
 'room': '--scale_rotation_scheduler --adam_eps_order 10 --freq_transmittance_threshold 0.8 --freq_opacity_threshold 0.5 --batch_size 2 --optimizer_type hybrid --sample_bbox_faces --warmup_densification --mult 0.25 --iterations 3000 --position_lr_max_steps 3000 --opacity_reset_interval 1200 --densification_interval 500 --densify_until_iter 1900 --densify_from_iter 100 --prune_iterations 1900',
 'bicycle': '--optimizer_type hybrid --split_ratio_threshold 0.4 --sample_bbox_faces --warmup_densification --mult 0.25 --iterations 3000 --position_lr_max_steps 3000 --opacity_reset_interval 1200 --densification_interval 500 --densify_until_iter 1900 --densify_from_iter 100 --prune_iterations 1900',
 'garden': '--adam_eps_order 10 --optimizer_type hybrid --split_ratio_threshold 0.4 --sample_bbox_faces --warmup_densification --mult 0.25 --iterations 3000 --position_lr_max_steps 3000 --opacity_reset_interval 1200 --densification_interval 500 --densify_until_iter 1900 --densify_from_iter 100 --prune_iterations 1900',
}

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def run(cmd, cwd, env, log_path, meta, gpu_sample_path=None):
    start = time.monotonic(); began = utc()
    meta['command'] = cmd; meta['cwd'] = str(cwd); meta['started_utc'] = began
    with open(log_path, 'w', encoding='utf-8') as log:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, text=True)
        while p.poll() is None:
            if time.monotonic() - start > 21600:
                os.killpg(p.pid, signal.SIGTERM); time.sleep(3)
                if p.poll() is None: os.killpg(p.pid, signal.SIGKILL)
                break
            if gpu_sample_path:
                out = subprocess.run(['nvidia-smi','--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits','-i','3'], capture_output=True, text=True, check=True).stdout.strip()
                with open(gpu_sample_path,'a',encoding='utf-8') as f: f.write(out+'\n')
                used = int(out.split(',')[3].strip())
                if used >= 73728:
                    os.killpg(p.pid, signal.SIGTERM); time.sleep(3)
                    if p.poll() is None: os.killpg(p.pid, signal.SIGKILL)
                    meta['stop_reason'] = 'GPU3 memory >= 73728 MiB'; break
            time.sleep(1)
        rc = p.wait()
    elapsed = time.monotonic() - start
    meta.update({'ended_utc':utc(),'wall_seconds':elapsed,'return_code':rc})
    if rc != 0: raise RuntimeError(f"{meta['name']} exited {rc}: {log_path}")

def tree_hash(root):
    h=hashlib.sha256(); files=sorted(p for p in root.rglob('*') if p.is_file())
    for f in files:
        h.update(f.relative_to(root).as_posix().encode()+b'\0'+hashlib.sha256(f.read_bytes()).hexdigest().encode()+b'\n')
    return h.hexdigest(),len(files)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gpu_state():
    q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits','-i','3'],capture_output=True,text=True,check=True).stdout.strip()
    procs=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader','-i','3'],capture_output=True,text=True,check=True).stdout.strip()
    return {'gpu':q,'compute_apps':procs}
def assert_preflight(scene):
    head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    status=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
    if head!='029814fb5ed773e4a794cec5115951a64f610ab8' or status: raise RuntimeError('source identity/cleanliness mismatch')
    ds,count=tree_hash(DATA/scene)
    if (ds,count)!=EXPECTED_DATA[scene]: raise RuntimeError(f'{scene} common data hash mismatch: {count} {ds}')
    sp=ART/f'expected_names_{scene}.json'
    if sha(sp)!=EXPECTED_SPLITS[scene]: raise RuntimeError(f'{scene} split manifest hash mismatch')
    s=gpu_state(); fields=[x.strip() for x in s['gpu'].split(',')]
    if fields[1]!='GPU-431ac351-20f8-bdf7-d798-1c9a3f545b72' or int(fields[2])>256 or int(fields[3])>10 or s['compute_apps']:
        raise RuntimeError(f'GPU3 preflight failed: {s}')
    out=ROOT/scene
    if out.exists(): raise RuntimeError(f'output already exists: {out}')
    out.mkdir(parents=True)
    return {'source_head':head,'source_status':status,'dataset_hash':ds,'dataset_files':count,
            'split_manifest_sha256':sha(sp),'gpu_before':s}

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    meta_path=ROOT/'execution.meta.json'
    if meta_path.exists():
        root_meta=json.loads(meta_path.read_text())
        root_meta['recovery_started_utc']=utc()
    else:
        root_meta={'protocol':'SAD_GS_SAME_LOSS_P16_TRIAD_20260928','concurrency_note':'P16 GPU2 profiles/room FastGS chain and GPU0/1 work were live; root explicitly authorized isolated GPU3 overlap. Wall comparisons remain descriptive.','scenes':{},'started_utc':utc()}
    meta_path.write_text(json.dumps(root_meta,indent=2)+'\n')
    common_env=os.environ.copy(); common_env.update({
       'CUDA_VISIBLE_DEVICES':'3','CUDA_HOME':'/usr/local/cuda-12.8','TORCH_CUDA_ARCH_LIST':'8.0',
       'PATH':'/root/miniforge3/envs/sadgs_baseline_clone_20260927/bin:/usr/local/cuda-12.8/bin:'+os.environ.get('PATH',''),
       'PYTHONDONTWRITEBYTECODE':'1',
       'PYTHONPATH':'/root/miniforge3/envs/sadgs_baseline_clone_20260927/lib/python3.11/site-packages:/root/miniforge3/envs/codex_nerficg/lib/python3.11/site-packages'})
    for scene in ('room','bicycle','garden'):
        out=ROOT/scene
        prior=out/'run.meta.json'
        if prior.exists():
            prior_doc=json.loads(prior.read_text())
            if prior_doc.get('status')=='complete_validated':
                root_meta['scenes'][scene]={'status':'complete_validated','metrics':prior_doc['metrics'],'training_wall_s':prior_doc['phases'][0]['wall_seconds']}
                continue
            raise RuntimeError(f'incomplete prior output exists; preserve and inspect: {out}')
        sm={'preflight':assert_preflight(scene),'phases':[]}
        (out/'gpu_samples.csv').write_text('timestamp,index,uuid,memory_mib,utilization_percent\n')
        # Same process/seed implementation; safe_state seeds Python, NumPy and Torch to 0.
        model=out/'model'; data=DATA/scene
        train=[PY,'train.py','-s',str(data),'-i','images','--eval','-r','1',*FLAGS[scene].split(),
               '--lambda_l2','0.0','--save_iterations','3000','--model_path',str(model)]
        env=common_env.copy(); env['OAR_JOB_ID']=f'sadgs_same_loss_p16_{scene}'
        phase={'name':'train'}; sm['phases'].append(phase)
        run(train,REPO,env,str(out/'train.log'),phase,str(out/'gpu_samples.csv'))
        if not (model/'point_cloud/iteration_3000/point_cloud.ply').exists(): raise RuntimeError(f'{scene}: final checkpoint absent')
        render=[PY,'render.py','-m',str(model),'--skip_train','--iteration','3000']
        phase={'name':'render'}; sm['phases'].append(phase)
        run(render,REPO,env,str(out/'render.log'),phase,str(out/'gpu_samples.csv'))
        rend=model/'test/ours_3000/renders'
        exp=json.loads((ART/f'expected_names_{scene}.json').read_text())
        names=[Path(x).stem for x in exp['test_names']]
        rendered=sorted(x.name for x in rend.glob('*.png'))
        expected_files=[f'{i:05d}.png' for i in range(len(names))]
        if rendered!=expected_files: raise RuntimeError(f'{scene}: render filenames/count mismatch')
        nmap=out/'name_map.json'
        metenv=os.environ.copy(); metenv['CUDA_VISIBLE_DEVICES']='3'
        map_cmd=[METPY,'/root/epic05_p16rb/scripts/p16_name_map.py','--method','SADGS_same_loss','--scene',scene,
                 '--rgb-dir',str(rend),'--gt-dir',str(rend.parent/'gt'),'--out',str(nmap)]
        phase={'name':'name_map'}; sm['phases'].append(phase)
        run(map_cmd,Path('/root/epic05_p16rb'),metenv,str(out/'name_map.log'),phase,str(out/'gpu_samples.csv'))
        map_doc=json.loads(nmap.read_text())
        if map_doc.get('status')!='PASS' or len(map_doc.get('name_map',{}))!=len(names): raise RuntimeError(f'{scene}: GT-pixel name map failed')
        metrics=[METPY,'/root/epic05_p16rb/scripts/p16_common_metrics.py','--method','SADGS_same_loss','--scene',scene,
                 '--rgb-dir',str(rend),'--out',str(out/'metrics_common.json'),'--gt-dir',str(data/'images'),
                 '--expected-names',str(ART/f'expected_names_{scene}.json'),'--name-map',str(nmap)]
        metenv=os.environ.copy(); metenv['CUDA_VISIBLE_DEVICES']='3'
        phase={'name':'common_metrics'}; sm['phases'].append(phase)
        run(metrics,Path('/root/epic05_p16rb'),metenv,str(out/'metrics.log'),phase,str(out/'gpu_samples.csv'))
        result=json.loads((out/'metrics_common.json').read_text())
        if result.get('status')!='SUCCESS' or result.get('n_expected')!=result.get('n_evaluated') or result.get('n_missing')!=0 or result.get('resolution_seen')!=[[1920,1080]]:
            raise RuntimeError(f'{scene}: common metric validation/resolution failed')
        sm.update({'status':'complete_validated','metrics':result['mean'],'n_expected':result['n_expected'],'n_evaluated':result['n_evaluated'],
                   'source_after_status':subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip(),
                   'gpu_after':gpu_state(),'ended_utc':utc()})
        (out/'run.meta.json').write_text(json.dumps(sm,indent=2)+'\n')
        root_meta['scenes'][scene]={'status':sm['status'],'metrics':sm['metrics'],'training_wall_s':sm['phases'][0]['wall_seconds']}
        (ROOT/'execution.meta.json').write_text(json.dumps(root_meta,indent=2)+'\n')
    root_meta['status']='complete_validated'; root_meta['ended_utc']=utc()
    (ROOT/'execution.meta.json').write_text(json.dumps(root_meta,indent=2)+'\n')

if __name__=='__main__':
    try: main()
    except Exception as e:
        print(f'ABORT: {e}',file=sys.stderr)
        raise





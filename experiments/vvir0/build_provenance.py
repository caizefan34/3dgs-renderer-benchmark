"""VRIR0 Phase P0 — freeze and provenance.

Records git state, baseline file hashes, installed gsplat source hash/path,
GPU/driver/CUDA/PyTorch, scene source hashes, frozen camera sequence hashes.
Emits outputs/P0_PROVENANCE.json. Fail-closed: refuses if required fields missing.
"""
import sys, os, json, hashlib, subprocess, time, platform
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'baseline', 'reference_v1'))
sys.path.insert(0, os.path.join(REPO, 'src'))
import vvir0_patch; vvir0_patch.apply()

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

def sha256_str(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def git_info():
    def run(*a):
        try:
            r = subprocess.run(['git'] + list(a), capture_output=True, text=True, timeout=15, cwd=REPO)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None
    commit = run('rev-parse', 'HEAD'); branch = run('rev-parse', '--abbrev-ref', 'HEAD')
    status = run('status', '--porcelain') or ''
    return commit, branch, bool(status.strip()), sha256_str(status)

def gsplat_info():
    import gsplat
    gdir = os.path.dirname(gsplat.__file__)
    ver = '1.4.0'
    try:
        import importlib.metadata as md
        ver = md.version('gsplat')
    except Exception:
        pass
    # hash the key installed source files
    files = ['rendering.py', 'cuda/csrc/isect_tiles.cu', 'cuda/csrc/rasterize_to_indices_in_range.cu',
             'cuda/csrc/fully_fused_projection_fwd.cu', 'version.py']
    hashes = {}
    for f in files:
        p = os.path.join(gdir, f.replace('/', os.sep))
        if os.path.exists(p):
            hashes[f] = sha256_file(p)
    return {'version': ver, 'path': gdir, 'source_hashes': hashes,
            'source_sha256_combined': sha256_str(json.dumps(hashes, sort_keys=True))}

def gpu_info():
    info = {'gpu_name': None, 'gpu_count': 0, 'cuda_runtime': torch.version.cuda,
            'capability': None, 'driver': None}
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_count'] = torch.cuda.device_count()
        info['capability'] = '.'.join(str(x) for x in torch.cuda.get_device_capability(0))
    try:
        info['driver'] = os.environ.get('NVIDIA_DRIVER_CAPABILITY') and torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info

def scene_hashes(scene):
    out = {}
    base = os.path.join(REPO, 'data', 'official', 'mipnerf360', scene)
    for rel in ['point_cloud.ply', 'cameras.json']:
        p = os.path.join(base, rel)
        if os.path.exists(p):
            out[rel] = sha256_file(p)
    sfm = os.path.join(REPO, 'data', 'datasets', 'mipnerf360', scene, 'sparse', '0', 'points3D.bin')
    if os.path.exists(sfm):
        out['sparse/0/points3D.bin'] = sha256_file(sfm)
    return out

def camera_seq_hash(scene, seed, n_cameras):
    import random, numpy as np
    rng = random.Random(seed)
    stack = list(range(n_cameras)); seq = []
    for _ in range(n_cameras):
        if not stack:
            stack = list(range(n_cameras))
        idx = rng.randint(0, len(stack) - 1)
        seq.append(stack.pop(idx))
    return sha256_str(json.dumps(seq)), seq

def main():
    scenes = ['room', 'bicycle', 'garden']
    commit, branch, dirty, diff_hash = git_info()
    gi = gpu_info(); si = gsplat_info()
    prov = {
        'experiment_id': f'vir0_{int(time.time())}',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'git_commit': commit, 'git_branch': branch, 'git_dirty': dirty,
        'git_diff_sha256': diff_hash,
        'baseline_dir': os.path.join(REPO, 'baseline', 'reference_v1'),
        'baseline_hashes': {
            'trainer.py': sha256_file(os.path.join(REPO, 'baseline', 'reference_v1', 'trainer.py')),
            'config.py': sha256_file(os.path.join(REPO, 'baseline', 'reference_v1', 'config.py')),
            'gaussian_model.py': sha256_file(os.path.join(REPO, 'baseline', 'reference_v1', 'gaussian_model.py')),
        },
        'gsplat': si,
        'torch_version': torch.__version__, 'python': platform.python_version(),
        'platform': platform.platform(), 'gpu': gi,
        'scenes': {},
        'seed': 42,
    }
    # camera counts + seq hashes
    sys.path.insert(0, os.path.join(REPO, 'scripts', 'epic05', 'phase7'))
    from dataset import GTDataset
    for sc in scenes:
        try:
            ds = GTDataset(scene=sc, repo_root=REPO, resolution='1080p', device='cpu', background='black')
            nc = len(ds)
            sh, seq = camera_seq_hash(sc, 42, nc)
            prov['scenes'][sc] = {'n_cameras': int(nc), 'camera_seq_sha256': sh,
                                  'source_hashes': scene_hashes(sc)}
        except Exception as e:
            prov['scenes'][sc] = {'error': repr(e)}
    outdir = os.path.join(REPO, 'experiments', 'vvir0', 'outputs')
    os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, 'P0_PROVENANCE.json')
    with open(outp, 'w') as f:
        json.dump(prov, f, indent=2)
    print('P0 provenance written:', outp)
    print(json.dumps({k: prov[k] for k in ['git_commit','git_branch','git_dirty','torch_version']}, indent=2))
    print('gsplat', si['version'], 'gpu', gi['gpu_name'], gi['capability'])
    for sc in scenes:
        print(sc, prov['scenes'][sc].get('n_cameras'), prov['scenes'][sc].get('camera_seq_sha256','')[:16])

if __name__ == '__main__':
    main()

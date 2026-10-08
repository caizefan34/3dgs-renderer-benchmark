import sys, os, hashlib, json, datetime
sys.path.insert(0, '/root/epic05_n1')
import n1_core as C

DPC_SO = '/root/.cache/torch_extensions/py312_cu128/dpc0_gaussian_render_inference_scene_cuda/dpc0_gaussian_render_inference_scene_cuda.so'

def bootstrap_dpc():
    os.environ['HIGS_PX_RUNTIME'] = '2'
    os.environ['HIGS_BWD_SCALAR_ADJOINT'] = 'scalar_adjoint'
    os.environ['HIGS_BWD_H8_MR'] = '1'
    os.environ['HIGS_BWD_ABSGRAD'] = '1'
    os.environ.pop('HIGS_DISABLE_F9', None)
    sys.path.insert(0, C.C0_WT)
    sys.path.insert(0, C.SCENE_CUDA_DIR)
    core = C.load_ext_module('gsplat_cuda', C.CORE_SO)
    sys.modules['gsplat.csrc'] = core
    exp = C.load_ext_module('experimental_gaussian_render_inference_scene_cuda', DPC_SO)
    sys.modules['gsplat.experimental.render.kernels.csrc'] = exp
    sys.modules['experimental_gaussian_render_inference_scene_cuda'] = exp
    sys.path.insert(0, C.PHASE7_DIR)
    sys.path.insert(0, C.SRC_DIR)
    sys.path.insert(0, C.REFERENCE_V1_DIR)
    from gsplat.experimental import rasterize_gaussian_higs_dynamic
    from gsplat.experimental.render.functional.gaussian_inference import _HIGS_DYNAMIC_SCENE
    return exp

exp = bootstrap_dpc()
print('DPC_SO loaded:', hasattr(exp, 'higs_rasterize_backward'))
print('DPC_SO sha256:', C.sha256_file(DPC_SO))

tag      = sys.argv[1] if len(sys.argv) > 1 else 'dpc'
n_repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 32
gpu      = int(sys.argv[3]) if len(sys.argv) > 3 else 0
scene    = sys.argv[4] if len(sys.argv) > 4 else 'room'
it       = int(sys.argv[5]) if len(sys.argv) > 5 else 10000
cam_idx  = int(sys.argv[6]) if len(sys.argv) > 6 else 5

torch.cuda.set_device(gpu)
torch.manual_seed(0)

ckpt = C.load_checkpoint(scene, it)
model = C.build_model(ckpt)
C.scene_reset()
ds = C.load_scene_dataset(scene)
cam, gt = ds.get_item(cam_idx)

img, _, _ = C.render_view(model, cam, model.active_sh_degree)
b = img.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
g = gt.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
fused = C.make_fused_loss()
L = fused(img, gt)
params = [getattr(model, n) for n in C.PARAM_NAMES]

GRAD_ORDER = ('_xyz', '_shs', '_scaling', '_rotation', '_opacity')

def hash_grads():
    h = hashlib.sha256()
    for name in GRAD_ORDER:
        t = getattr(model, name).grad
        if t is None:
            h.update(b'NULL'); continue
        h.update(t.detach().to('cpu', torch.float32).numpy().tobytes())
    return h.hexdigest()

torch.autograd.grad(L, params, retain_graph=True)
model.optimizer.zero_grad(set_to_none=True)

hashes = []
for r in range(n_repeat):
    grads = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    for p, gr in zip(params, grads):
        if gr is None:
            p.grad = None
        else:
            p.grad = gr.detach().clone()
    hashes.append(hash_grads())

uniq = sorted(set(hashes))
out = {
    'tag': tag, 'gpu': gpu, 'scene': scene, 'iter': it, 'cam': cam_idx,
    'n_repeat': n_repeat, 'n_unique_hashes': len(uniq),
    'deterministic': len(uniq) == 1,
    'first_hash': hashes[0],
    'dpc_so_sha256': C.sha256_file(DPC_SO),
    'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
}
print(json.dumps({k: v for k, v in out.items() if k != 'all_hashes'}, indent=2))
os.makedirs('/root/epic05_dpc0/runs', exist_ok=True)
with open(f'/root/epic05_dpc0/runs/{tag}_gpu{gpu}_{scene}_{it}_cam{cam_idx}.json', 'w') as f:
    json.dump(out, f, indent=2)